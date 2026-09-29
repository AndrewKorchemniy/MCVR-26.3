#pragma once

#include "core/all_extern.hpp"

#include <memory>

namespace vk {

enum class WindowSurfaceBackend {
    Glfw,
    Xlib,
    Wayland,
    Win32,
};

class PresentationTarget {
  public:
    virtual ~PresentationTarget() = default;

    virtual WindowSurfaceBackend surfaceBackend() const = 0;
    virtual GLFWwindow *sourceWindow() const = 0;
    virtual VkResult createSurface(VkInstance instance, VkSurfaceKHR *surface) = 0;
    virtual void syncSize() = 0;
};

WindowSurfaceBackend selectNativeWindowSurfaceBackend();
const char *windowSurfaceBackendName(WindowSurfaceBackend backend);
const char *requiredWindowSurfaceExtension(WindowSurfaceBackend backend);
VkResult
createWindowSurface(WindowSurfaceBackend backend, VkInstance instance, GLFWwindow *window, VkSurfaceKHR *surface);
std::shared_ptr<PresentationTarget>
createExistingWindowPresentationTarget(GLFWwindow *window, WindowSurfaceBackend backend);
std::shared_ptr<PresentationTarget> createOpenGLUiPresentationTarget(GLFWwindow *window);

}
