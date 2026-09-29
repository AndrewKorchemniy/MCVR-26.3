#include "core/vulkan/window_surface.hpp"

#include <mutex>
#include <stdexcept>
#include <string>

namespace {
class ExistingWindowPresentationTarget final : public vk::PresentationTarget {
  public:
    ExistingWindowPresentationTarget(GLFWwindow *window, vk::WindowSurfaceBackend backend)
        : window_(window), backend_(backend) {
        if (window_ == nullptr) { throw std::invalid_argument("Presentation target requires a GLFW window"); }
    }

    vk::WindowSurfaceBackend surfaceBackend() const override {
        return backend_;
    }

    GLFWwindow *sourceWindow() const override {
        return window_;
    }

    VkResult createSurface(VkInstance instance, VkSurfaceKHR *surface) override {
        return vk::createWindowSurface(backend_, instance, window_, surface);
    }

    void syncSize() override {}

  private:
    GLFWwindow *window_;
    vk::WindowSurfaceBackend backend_;
};

#if defined(__linux__) && defined(MCVR_UI_BACKEND_OPENGL)
class XlibChildPresentationTarget final : public vk::PresentationTarget {
  public:
    explicit XlibChildPresentationTarget(GLFWwindow *window) : sourceWindow_(window) {
        if (window == nullptr || GLFW_GetX11Display == nullptr || GLFW_GetX11Window == nullptr) {
            throw std::invalid_argument("Xlib presentation target requires a valid GLFW window");
        }
        display_ = GLFW_GetX11Display();
        parentWindow_ = GLFW_GetX11Window(window);
        if (display_ == nullptr || parentWindow_ == 0) {
            throw std::runtime_error("GLFW did not provide a valid X11 display and window");
        }
        XWindowAttributes attributes{};
        if (!XGetWindowAttributes(display_, parentWindow_, &attributes)) {
            throw std::runtime_error("Failed to query the OpenGL window client area");
        }
        childWindow_ = XCreateSimpleWindow(display_, parentWindow_, 0, 0,
                                          attributes.width > 0 ? attributes.width : 1,
                                          attributes.height > 0 ? attributes.height : 1, 0, 0, 0);
        if (childWindow_ == 0) { throw std::runtime_error("Failed to create the Xlib Vulkan child window"); }
        XMapWindow(display_, childWindow_);
        XSync(display_, False);
    }

    ~XlibChildPresentationTarget() override {
        if (childWindow_ != 0) {
            XDestroyWindow(display_, childWindow_);
            XFlush(display_);
        }
    }

    vk::WindowSurfaceBackend surfaceBackend() const override { return vk::WindowSurfaceBackend::Xlib; }
    GLFWwindow *sourceWindow() const override { return sourceWindow_; }

    VkResult createSurface(VkInstance instance, VkSurfaceKHR *surface) override {
        if (surface == nullptr) { return VK_ERROR_INITIALIZATION_FAILED; }
        auto createXlibSurface = reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(
            vkGetInstanceProcAddr(instance, "vkCreateXlibSurfaceKHR"));
        if (createXlibSurface == nullptr) {
            throw std::runtime_error("Vulkan instance does not expose vkCreateXlibSurfaceKHR");
        }
        VkXlibSurfaceCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        createInfo.dpy = display_;
        createInfo.window = childWindow_;
        return createXlibSurface(instance, &createInfo, nullptr, surface);
    }

    void syncSize() override {
        XWindowAttributes attributes{};
        if (!XGetWindowAttributes(display_, parentWindow_, &attributes)) {
            throw std::runtime_error("Failed to query the OpenGL window client area during resize");
        }
        ::Window replacement = XCreateSimpleWindow(display_, parentWindow_, 0, 0,
                                                     attributes.width > 0 ? attributes.width : 1,
                                                     attributes.height > 0 ? attributes.height : 1, 0, 0, 0);
        if (replacement == 0) { throw std::runtime_error("Failed to recreate the Xlib Vulkan child window"); }
        XDestroyWindow(display_, childWindow_);
        childWindow_ = replacement;
        XMapWindow(display_, childWindow_);
        XSync(display_, False);
    }

  private:
    GLFWwindow *sourceWindow_ = nullptr;
    Display *display_ = nullptr;
    ::Window parentWindow_ = 0;
    ::Window childWindow_ = 0;
};
#endif

#if defined(_WIN32)
constexpr wchar_t presentationWindowClassName[] = L"RadianceVulkanPresentationTarget";

LRESULT CALLBACK presentationWindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    if (message == WM_NCHITTEST) { return HTTRANSPARENT; }
    if (message == WM_ERASEBKGND) { return 1; }
    return DefWindowProcW(window, message, wParam, lParam);
}

void registerPresentationWindowClass() {
    static std::once_flag registrationFlag;
    std::call_once(registrationFlag, [] {
        WNDCLASSEXW windowClass{};
        windowClass.cbSize = sizeof(windowClass);
        windowClass.lpfnWndProc = presentationWindowProc;
        windowClass.hInstance = GetModuleHandleW(nullptr);
        windowClass.lpszClassName = presentationWindowClassName;
        if (RegisterClassExW(&windowClass) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            throw std::runtime_error("Failed to register the Win32 Vulkan presentation window class");
        }
    });
}

class Win32ChildPresentationTarget final : public vk::PresentationTarget {
  public:
    explicit Win32ChildPresentationTarget(GLFWwindow *window) : sourceWindow_(window) {
        if (sourceWindow_ == nullptr || GLFW_GetWin32Window == nullptr) {
            throw std::invalid_argument("Win32 presentation target requires a valid GLFW window");
        }

        parentWindow_ = GLFW_GetWin32Window(sourceWindow_);
        if (parentWindow_ == nullptr) { throw std::runtime_error("GLFW did not provide a valid parent HWND"); }

        registerPresentationWindowClass();
        RECT clientRect{};
        if (!GetClientRect(parentWindow_, &clientRect)) {
            throw std::runtime_error("Failed to query the OpenGL window client area");
        }

        childWindow_ = CreateWindowExW(
            WS_EX_NOACTIVATE | WS_EX_NOPARENTNOTIFY, presentationWindowClassName, L"",
            WS_CHILD | WS_VISIBLE | WS_DISABLED | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, 0, 0,
            clientRect.right - clientRect.left, clientRect.bottom - clientRect.top, parentWindow_, nullptr,
            GetModuleHandleW(nullptr), nullptr);
        if (childWindow_ == nullptr) {
            throw std::runtime_error("Failed to create the Win32 Vulkan child presentation window");
        }
    }

    ~Win32ChildPresentationTarget() override {
        if (childWindow_ == nullptr || !IsWindow(childWindow_)) return;

        DWORD ownerThread = GetWindowThreadProcessId(childWindow_, nullptr);
        if (ownerThread == GetCurrentThreadId()) {
            DestroyWindow(childWindow_);
        } else {
            PostMessageW(childWindow_, WM_CLOSE, 0, 0);
        }
    }

    vk::WindowSurfaceBackend surfaceBackend() const override {
        return vk::WindowSurfaceBackend::Win32;
    }

    GLFWwindow *sourceWindow() const override {
        return sourceWindow_;
    }

    VkResult createSurface(VkInstance instance, VkSurfaceKHR *surface) override {
        if (surface == nullptr || childWindow_ == nullptr || !IsWindow(childWindow_)) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        auto createWin32Surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
            vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR"));
        if (createWin32Surface == nullptr) {
            throw std::runtime_error("Vulkan instance does not expose vkCreateWin32SurfaceKHR");
        }

        VkWin32SurfaceCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.hinstance = GetModuleHandleW(nullptr);
        createInfo.hwnd = childWindow_;
        return createWin32Surface(instance, &createInfo, nullptr, surface);
    }

    void syncSize() override {
        if (childWindow_ == nullptr || !IsWindow(childWindow_)) {
            throw std::runtime_error("Win32 Vulkan child presentation window is no longer valid");
        }
        RECT clientRect{};
        if (!GetClientRect(parentWindow_, &clientRect)) {
            throw std::runtime_error("Failed to query the OpenGL window client area during resize");
        }
        if (!SetWindowPos(childWindow_, HWND_TOP, 0, 0, clientRect.right - clientRect.left,
                          clientRect.bottom - clientRect.top, SWP_NOACTIVATE | SWP_SHOWWINDOW)) {
            throw std::runtime_error("Failed to resize the Win32 Vulkan child presentation window");
        }
    }

  private:
    GLFWwindow *sourceWindow_ = nullptr;
    HWND parentWindow_ = nullptr;
    HWND childWindow_ = nullptr;
};
#endif
}

vk::WindowSurfaceBackend vk::selectNativeWindowSurfaceBackend() {
    if (GLFW_GetPlatform == nullptr) { throw std::runtime_error("Loaded GLFW does not expose glfwGetPlatform"); }

    const int platform = GLFW_GetPlatform();
#if defined(_WIN32)
    if (platform == GLFW_PLATFORM_WIN32) { return WindowSurfaceBackend::Win32; }
#elif defined(__linux__) || defined(__unix__)
    if (platform == GLFW_PLATFORM_X11) { return WindowSurfaceBackend::Xlib; }
    if (platform == GLFW_PLATFORM_WAYLAND) { return WindowSurfaceBackend::Wayland; }
#endif

    throw std::runtime_error("OpenGL UI native Vulkan surface does not support GLFW platform " +
                             std::to_string(platform));
}

const char *vk::windowSurfaceBackendName(WindowSurfaceBackend backend) {
    switch (backend) {
        case WindowSurfaceBackend::Glfw: return "GLFW";
        case WindowSurfaceBackend::Xlib: return "Xlib";
        case WindowSurfaceBackend::Wayland: return "Wayland";
        case WindowSurfaceBackend::Win32: return "Win32";
    }
    return "unknown";
}

const char *vk::requiredWindowSurfaceExtension(WindowSurfaceBackend backend) {
    switch (backend) {
        case WindowSurfaceBackend::Glfw: return nullptr;
#if defined(__linux__) || defined(__unix__)
        case WindowSurfaceBackend::Xlib: return VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
        case WindowSurfaceBackend::Wayland: return VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;
#endif
#if defined(_WIN32)
        case WindowSurfaceBackend::Win32: return VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
#endif
        default: throw std::runtime_error("Window surface backend is not available in this build");
    }
}

VkResult
vk::createWindowSurface(WindowSurfaceBackend backend, VkInstance instance, GLFWwindow *window, VkSurfaceKHR *surface) {
    if (window == nullptr || surface == nullptr) { return VK_ERROR_INITIALIZATION_FAILED; }

    switch (backend) {
        case WindowSurfaceBackend::Glfw: return GLFW_CreateWindowSurface(instance, window, nullptr, surface);
#if defined(__linux__) || defined(__unix__)
        case WindowSurfaceBackend::Xlib: {
            if (GLFW_GetX11Display == nullptr || GLFW_GetX11Window == nullptr) {
                throw std::runtime_error("Xlib Vulkan surface functions are unavailable");
            }
            auto createXlibSurface =
                reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(vkGetInstanceProcAddr(instance, "vkCreateXlibSurfaceKHR"));
            if (createXlibSurface == nullptr) {
                throw std::runtime_error("Vulkan instance does not expose vkCreateXlibSurfaceKHR");
            }
            Display *display = GLFW_GetX11Display();
            ::Window nativeWindow = GLFW_GetX11Window(window);
            if (display == nullptr || nativeWindow == 0) {
                throw std::runtime_error("GLFW did not provide a valid X11 display and window");
            }

            VkXlibSurfaceCreateInfoKHR createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
            createInfo.dpy = display;
            createInfo.window = nativeWindow;
            return createXlibSurface(instance, &createInfo, nullptr, surface);
        }
        case WindowSurfaceBackend::Wayland: {
            if (GLFW_GetWaylandDisplay == nullptr || GLFW_GetWaylandWindow == nullptr) {
                throw std::runtime_error("Wayland Vulkan surface functions are unavailable");
            }
            auto createWaylandSurface = reinterpret_cast<PFN_vkCreateWaylandSurfaceKHR>(
                vkGetInstanceProcAddr(instance, "vkCreateWaylandSurfaceKHR"));
            if (createWaylandSurface == nullptr) {
                throw std::runtime_error("Vulkan instance does not expose vkCreateWaylandSurfaceKHR");
            }
            wl_display *display = GLFW_GetWaylandDisplay();
            wl_surface *nativeWindow = GLFW_GetWaylandWindow(window);
            if (display == nullptr || nativeWindow == nullptr) {
                throw std::runtime_error("GLFW did not provide a valid Wayland display and surface");
            }

            VkWaylandSurfaceCreateInfoKHR createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
            createInfo.display = display;
            createInfo.surface = nativeWindow;
            return createWaylandSurface(instance, &createInfo, nullptr, surface);
        }
#endif
#if defined(_WIN32)
        case WindowSurfaceBackend::Win32: {
            if (GLFW_GetWin32Window == nullptr) {
                throw std::runtime_error("Win32 Vulkan surface functions are unavailable");
            }
            auto createWin32Surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
                vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR"));
            if (createWin32Surface == nullptr) {
                throw std::runtime_error("Vulkan instance does not expose vkCreateWin32SurfaceKHR");
            }
            HWND nativeWindow = GLFW_GetWin32Window(window);
            if (nativeWindow == nullptr) { throw std::runtime_error("GLFW did not provide a valid Win32 window"); }

            VkWin32SurfaceCreateInfoKHR createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
            createInfo.hinstance = GetModuleHandleW(nullptr);
            createInfo.hwnd = nativeWindow;
            return createWin32Surface(instance, &createInfo, nullptr, surface);
        }
#endif
        default: throw std::runtime_error("Window surface backend is not available in this build");
    }
}

std::shared_ptr<vk::PresentationTarget>
vk::createExistingWindowPresentationTarget(GLFWwindow *window, WindowSurfaceBackend backend) {
    return std::make_shared<ExistingWindowPresentationTarget>(window, backend);
}

std::shared_ptr<vk::PresentationTarget> vk::createOpenGLUiPresentationTarget(GLFWwindow *window) {
    WindowSurfaceBackend backend = selectNativeWindowSurfaceBackend();
#if defined(__linux__) && defined(MCVR_UI_BACKEND_OPENGL)
    if (backend == WindowSurfaceBackend::Xlib) {
        return std::make_shared<XlibChildPresentationTarget>(window);
    }
#endif
#if defined(_WIN32)
    if (backend == WindowSurfaceBackend::Win32) {
        return std::make_shared<Win32ChildPresentationTarget>(window);
    }
#endif

    return createExistingWindowPresentationTarget(window, backend);
}
