#include "core/vulkan/window.hpp"

#include "core/vulkan/instance.hpp"
#include "core/vulkan/result.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

bool vk::Window::framebufferResized = false;

vk::Window::Window(std::shared_ptr<Instance> instance, uint32_t width, uint32_t height)
    : instance_(instance), width_(width), height_(height) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window_ = glfwCreateWindow(width_, height_, "Vulkan Window", nullptr, nullptr);
    if (!window_) {
        std::cerr << "Cannot create glfw window!" << std::endl;
        GLFW_Terminate();
        exit(EXIT_FAILURE);
    }

    presentationTarget_ = createExistingWindowPresentationTarget(window_, surfaceBackend_);
    VkResult result = presentationTarget_->createSurface(instance_->vkInstance(), &surface_);
    if (result != VK_SUCCESS) {
        std::cerr << "Cannot create vulkan window surface!" << std::endl;
        GLFW_Terminate();
        exit(EXIT_FAILURE);
    }
}

vk::Window::Window(std::shared_ptr<Instance> instance,
                   GLFWwindow *window_,
                   WindowSurfaceBackend surfaceBackend)
    : Window(instance, createExistingWindowPresentationTarget(window_, surfaceBackend)) {}

vk::Window::Window(std::shared_ptr<Instance> instance, std::shared_ptr<PresentationTarget> presentationTarget)
    : instance_(instance), presentationTarget_(std::move(presentationTarget)) {
    if (presentationTarget_ == nullptr) { throw std::invalid_argument("Vulkan window requires a presentation target"); }

    window_ = presentationTarget_->sourceWindow();
    surfaceBackend_ = presentationTarget_->surfaceBackend();
    GLFW_GetWindowSize(window_, reinterpret_cast<int *>(&width_), reinterpret_cast<int *>(&height_));

    VkResult result = presentationTarget_->createSurface(instance_->vkInstance(), &surface_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string("Cannot create Vulkan window surface with ") +
                                 windowSurfaceBackendName(surfaceBackend_) + ": " + vk::resultName(result));
    }
}

vk::Window::~Window() {
    if (surface_ != VK_NULL_HANDLE) { vkDestroySurfaceKHR(instance_->vkInstance(), surface_, nullptr); }
}

uint32_t vk::Window::width() {
    return width_;
}

uint32_t vk::Window::height() {
    return height_;
}

GLFWwindow *vk::Window::window() {
    return window_;
}

VkSurfaceKHR &vk::Window::vkSurface() {
    return surface_;
}

void vk::Window::reconstructSurface() {
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_->vkInstance(), surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }

    presentationTarget_->syncSize();
    VkResult result = presentationTarget_->createSurface(instance_->vkInstance(), &surface_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string("Cannot recreate Vulkan window surface with ") +
                                 windowSurfaceBackendName(surfaceBackend_) + ": " + vk::resultName(result));
    }
}
