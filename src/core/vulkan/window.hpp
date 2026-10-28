#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Instance;

// Non-owning wrapper around the VkSurfaceKHR vanilla created through SDL. Never destroys the surface.
class Window : public SharedObject<Window> {
  public:
    Window(std::shared_ptr<Instance> instance, VkSurfaceKHR surface);
    ~Window();

    VkSurfaceKHR &vkSurface();

  private:
    std::shared_ptr<Instance> instance_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
};
}
