#pragma once

#include "core/all_extern.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/sync.hpp"

namespace vk {
class Device;
class PhysicalDevice;
}

namespace mcvr {

struct OpenGLUiSurfaceExport {
    uint32_t generation = 0;
    uint32_t frameIndex = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t memorySize = 0;
    uint64_t memoryHandle = 0;
    uint64_t vulkanReadySemaphoreHandle = 0;
    uint64_t glCompleteSemaphoreHandle = 0;
};

class OpenGLUiSurface : public vk::Image, public SharedObject<OpenGLUiSurface> {
  public:
    OpenGLUiSurface(std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                    std::shared_ptr<vk::Device> device,
                    uint32_t generation,
                    uint32_t frameIndex,
                    uint32_t width,
                    uint32_t height);
    ~OpenGLUiSurface();

    uint32_t generation() const;
    uint32_t frameIndex() const;
    uint64_t memorySize() const;
    VkImageLayout &imageLayout() override;
    VkImageSubresourceRange fullSubresourceRange() const override;
    VkSemaphore vulkanReadySemaphore() const;
    VkSemaphore glCompleteSemaphore() const;
    OpenGLUiSurfaceExport exportHandles();
    bool ownedByExternal() const;
    void setOwnedByExternal(bool ownedByExternal);

    uint32_t width() override;
    uint32_t height() override;
    uint32_t depth() override;
    uint32_t layer() override;
    VkFormat &vkFormat() override;
    VkImage &vkImage() override;
    VkImageView &vkImageView(int index = 0) override;

  private:
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const;
    void verifyExternalImageSupport() const;

    std::shared_ptr<vk::PhysicalDevice> physicalDevice_;
    std::shared_ptr<vk::Device> device_;
    uint32_t generation_;
    uint32_t frameIndex_;
    uint32_t width_;
    uint32_t height_;
    uint32_t depth_ = 1;
    uint32_t layer_ = 1;
    uint64_t memorySize_ = 0;
    VkFormat format_ = VK_FORMAT_R8G8B8A8_UNORM;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkImageLayout imageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    bool ownedByExternal_ = false;
    std::shared_ptr<vk::Semaphore> vulkanReadySemaphore_;
    std::shared_ptr<vk::Semaphore> glCompleteSemaphore_;
};

}
