#pragma once

#include "core/all_extern.hpp"

#include <vector>

namespace vk {
class PhysicalDevice;
class Device;
class Window;
class SwapchainImage;

class Swapchain : public SharedObject<Swapchain> {
    friend class SwapchainImage;
    friend class std::vector<SwapchainImage>;

  public:
    // The surface and device are vanilla's; a desired width and height of 0 mean "use the surface's current extent".
    Swapchain(std::shared_ptr<PhysicalDevice> physicalDevice,
              std::shared_ptr<Device> device,
              std::shared_ptr<Window> window);
    ~Swapchain();

    // Remembers the size and present mode the next reconstruct() creates the swapchain with.
    void setDesiredConfiguration(uint32_t width, uint32_t height, VkPresentModeKHR presentMode);
    bool matchesDesiredConfiguration();
    void reconstruct();
    VkSwapchainKHR &vkSwapchain();
    VkExtent2D &vkExtent();
    VkExtent2D &vkMaxExtent();
    VkExtent2D &vkMinExtent();
    VkSurfaceFormatKHR &vkSurfaceFormat();
    VkPresentModeKHR vkPresentMode() const;
    std::vector<std::shared_ptr<SwapchainImage>> &swapchainImages();
    uint32_t imageCount();

  private:
    std::shared_ptr<PhysicalDevice> physicalDevice_;
    std::shared_ptr<Device> device_;
    std::shared_ptr<Window> window_;

    uint32_t desiredWidth_ = 0;
    uint32_t desiredHeight_ = 0;
    VkPresentModeKHR desiredPresentMode_ = VK_PRESENT_MODE_FIFO_KHR;

    uint32_t imageCount_;
    VkSurfaceFormatKHR surfaceFormat_;
    VkPresentModeKHR presentMode_;
    VkExtent2D extent_;
    VkExtent2D maxExtent_;
    VkExtent2D minExtent_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;

    std::vector<std::shared_ptr<SwapchainImage>> swapchainImages_;
};
};
