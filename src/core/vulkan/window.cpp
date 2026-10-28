#include "core/vulkan/window.hpp"

#include "core/vulkan/instance.hpp"

#include <stdexcept>

vk::Window::Window(std::shared_ptr<Instance> instance, VkSurfaceKHR surface)
    : instance_(std::move(instance)), surface_(surface) {
    if (surface_ == VK_NULL_HANDLE) { throw std::invalid_argument("Vulkan surface handle is null"); }
}

vk::Window::~Window() {
    // The surface belongs to Minecraft, which destroys it after the swapchain is gone.
}

VkSurfaceKHR &vk::Window::vkSurface() {
    return surface_;
}
