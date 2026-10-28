#include "core/vulkan/swapchain.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/result.hpp"
#include "core/vulkan/window.hpp"

#include "core/render/renderer.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

std::ostream &swapchainCerr() {
    return std::cerr << "[Swapchain] ";
}

vk::Swapchain::Swapchain(std::shared_ptr<PhysicalDevice> physicalDevice,
                         std::shared_ptr<Device> device,
                         std::shared_ptr<Window> window)
    : physicalDevice_(physicalDevice), device_(device), window_(window) {
    reconstruct();
}

VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
        return {VK_FORMAT_R8G8B8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR};
    }

    auto formatRank = [](VkFormat format) {
        switch (format) {
            case VK_FORMAT_R8G8B8A8_UNORM: return 0;
            case VK_FORMAT_B8G8R8A8_UNORM: return 1;
            case VK_FORMAT_R8G8B8A8_SRGB: return 2;
            case VK_FORMAT_B8G8R8A8_SRGB: return 3;
            default: return 4;
        }
    };

    auto selectedFormat = std::min_element(
        availableFormats.begin(), availableFormats.end(),
        [&formatRank](const auto &lhs, const auto &rhs) { return formatRank(lhs.format) < formatRank(rhs.format); });

    if (selectedFormat->format == VK_FORMAT_R8G8B8A8_SRGB || selectedFormat->format == VK_FORMAT_B8G8R8A8_SRGB) {
        swapchainCerr() << "warning: selected SRGB surface format (" << selectedFormat->format
                        << "), UNORM format unavailable, this may cause color space error" << std::endl;
    }

    return *selectedFormat;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &surfaceCapabilities, uint32_t width, uint32_t height) {
    if (surfaceCapabilities.currentExtent.width == 0xFFFFFFFFu) {
        VkExtent2D swapChainExtent = {};

        swapChainExtent.width = std::min(std::max(width, surfaceCapabilities.minImageExtent.width),
                                         surfaceCapabilities.maxImageExtent.width);
        swapChainExtent.height = std::min(std::max(height, surfaceCapabilities.minImageExtent.height),
                                          surfaceCapabilities.maxImageExtent.height);
        return swapChainExtent;
    } else {
        return surfaceCapabilities.currentExtent;
    }
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR> presentModes, VkPresentModeKHR desired) {
    for (const auto &presentMode : presentModes) {
        if (presentMode == desired) { return presentMode; }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

void vk::Swapchain::setDesiredConfiguration(uint32_t width, uint32_t height, VkPresentModeKHR presentMode) {
    desiredWidth_ = width;
    desiredHeight_ = height;
    desiredPresentMode_ = presentMode;
}

bool vk::Swapchain::matchesDesiredConfiguration() {
    if (swapchain_ == VK_NULL_HANDLE) { return false; }
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &surfaceCapabilities) != VK_SUCCESS) {
        return false;
    }
    VkExtent2D wanted = chooseSwapExtent(surfaceCapabilities, desiredWidth_ == 0 ? 854 : desiredWidth_,
                                         desiredHeight_ == 0 ? 480 : desiredHeight_);
    return wanted.width == extent_.width && wanted.height == extent_.height && desiredPresentMode_ == presentMode_;
}

void vk::Swapchain::reconstruct() {
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &surfaceCapabilities) != VK_SUCCESS) {
        throw std::runtime_error("failed to acquire presentation surface capabilities");
    }

    maxExtent_ = surfaceCapabilities.maxImageExtent;
    minExtent_ = surfaceCapabilities.minImageExtent;

    imageCount_ = surfaceCapabilities.minImageCount + 1;
    imageCount_ = std::clamp(imageCount_, (uint32_t)2, (uint32_t)3);
    if (surfaceCapabilities.maxImageCount != 0 && imageCount_ > surfaceCapabilities.maxImageCount) {
        imageCount_ = surfaceCapabilities.maxImageCount;
    }

    uint32_t formatCount;
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(), &formatCount,
                                             nullptr) != VK_SUCCESS ||
        formatCount == 0) {
        throw std::runtime_error("failed to get number of supported surface formats");
    }

    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(), &formatCount,
                                             surfaceFormats.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to get supported surface formats");
    }

    surfaceFormat_ = chooseSurfaceFormat(surfaceFormats);

    uint32_t presentModeCount;
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &presentModeCount, nullptr) != VK_SUCCESS ||
        presentModeCount == 0) {
        throw std::runtime_error("failed to get number of supported presentation modes");
    }

    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &presentModeCount, presentModes.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to get supported presentation modes");
    }

    presentMode_ = choosePresentMode(presentModes, desiredPresentMode_);

    extent_ = chooseSwapExtent(surfaceCapabilities, desiredWidth_ == 0 ? 854 : desiredWidth_,
                               desiredHeight_ == 0 ? 480 : desiredHeight_);
    if (extent_.width == 0 || extent_.height == 0) { throw std::runtime_error("the surface has a zero-sized extent"); }

    VkSurfaceTransformFlagBitsKHR surfaceTransform;
    if (surfaceCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        surfaceTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else {
        surfaceTransform = surfaceCapabilities.currentTransform;
    }

    VkSwapchainKHR oldSwapchain = swapchain_;
    swapchainImages_.clear();

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = window_->vkSurface();
    createInfo.minImageCount = imageCount_;
    createInfo.imageFormat = surfaceFormat_.format;
    createInfo.imageColorSpace = surfaceFormat_.colorSpace;
    createInfo.imageExtent = extent_;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;
    createInfo.preTransform = surfaceTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode_;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = oldSwapchain;

    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    VkResult createResult = vkCreateSwapchainKHR(device_->vkDevice(), &createInfo, nullptr, &newSwapchain);
    if (createResult != VK_SUCCESS) {
        vk::requireSuccess(createResult, "vkCreateSwapchainKHR");
    }

    if (oldSwapchain != VK_NULL_HANDLE) { vkDestroySwapchainKHR(device_->vkDevice(), oldSwapchain, nullptr); }
    swapchain_ = newSwapchain;

    uint32_t actualImageCount = 0;
    if (vkGetSwapchainImagesKHR(device_->vkDevice(), swapchain_, &actualImageCount, nullptr) != VK_SUCCESS ||
        actualImageCount == 0) {
        throw std::runtime_error("failed to acquire number of swap chain images");
    }
    imageCount_ = actualImageCount;

    std::vector<VkImage> images(actualImageCount);
    if (vkGetSwapchainImagesKHR(device_->vkDevice(), swapchain_, &actualImageCount, images.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to acquire swap chain images");
    }
    for (int i = 0; i < actualImageCount; i++) {
        swapchainImages_.push_back(
            SwapchainImage::create(device_, images[i], extent_.width, extent_.height, surfaceFormat_.format));
    }
}

vk::Swapchain::~Swapchain() {
    swapchainImages_.clear();
    if (swapchain_ != VK_NULL_HANDLE) { vkDestroySwapchainKHR(device_->vkDevice(), swapchain_, nullptr); }
}

VkSwapchainKHR &vk::Swapchain::vkSwapchain() {
    return swapchain_;
}

VkExtent2D &vk::Swapchain::vkExtent() {
    return extent_;
}

VkExtent2D &vk::Swapchain::vkMaxExtent() {
    return maxExtent_;
}

VkExtent2D &vk::Swapchain::vkMinExtent() {
    return minExtent_;
}

VkSurfaceFormatKHR &vk::Swapchain::vkSurfaceFormat() {
    return surfaceFormat_;
}

VkPresentModeKHR vk::Swapchain::vkPresentMode() const {
    return presentMode_;
}

std::vector<std::shared_ptr<vk::SwapchainImage>> &vk::Swapchain::swapchainImages() {
    return swapchainImages_;
}

uint32_t vk::Swapchain::imageCount() {
    return imageCount_;
}
