#pragma once

#include "core/all_extern.hpp"
#include "core/vulkan/resource_memory_tracker.hpp"

#include <vector>

namespace vk {
class VMA;
class Device;
class Swapchain;
class Buffer;
class CommandBuffer;

size_t formatToByte(VkFormat format);

static VkImageSubresourceRange wholeColorSubresourceRange = {
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = VK_REMAINING_MIP_LEVELS,
    .baseArrayLayer = 0,
    .layerCount = VK_REMAINING_ARRAY_LAYERS,
};

static VkImageSubresourceRange wholeDepthSubresourceRange = {
    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
    .baseMipLevel = 0,
    .levelCount = VK_REMAINING_MIP_LEVELS,
    .baseArrayLayer = 0,
    .layerCount = VK_REMAINING_ARRAY_LAYERS,
};

static VkImageSubresourceRange wholeStencilSubresourceRange = {
    .aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT,
    .baseMipLevel = 0,
    .levelCount = VK_REMAINING_MIP_LEVELS,
    .baseArrayLayer = 0,
    .layerCount = VK_REMAINING_ARRAY_LAYERS,
};

static VkImageSubresourceRange wholeDepthStencilSubresourceRange = {
    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
    .baseMipLevel = 0,
    .levelCount = VK_REMAINING_MIP_LEVELS,
    .baseArrayLayer = 0,
    .layerCount = VK_REMAINING_ARRAY_LAYERS,
};

static VkImageSubresourceLayers wholeColorSubresourceLayers = {
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .mipLevel = 0,
    .baseArrayLayer = 0,
    .layerCount = VK_REMAINING_ARRAY_LAYERS,
};

static VkImageSubresourceLayers wholeDepthSubresourceLayers = {
    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
    .mipLevel = 0,
    .baseArrayLayer = 0,
    .layerCount = VK_REMAINING_ARRAY_LAYERS,
};

static VkImageSubresourceLayers wholeStencilSubresourceLayers = {
    .aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT,
    .mipLevel = 0,
    .baseArrayLayer = 0,
    .layerCount = VK_REMAINING_ARRAY_LAYERS,
};

class Image {
  public:
    virtual ~Image() = default;

    virtual uint32_t width() = 0;
    virtual uint32_t height() = 0;
    virtual uint32_t depth() = 0;
    virtual uint32_t layer() = 0;
    virtual VkFormat &vkFormat() = 0;
    virtual VkImage &vkImage() = 0;
    virtual VkImageView &vkImageView(int index = 0) = 0;
    virtual VkImageLayout &imageLayout() = 0;
    virtual VkImageSubresourceRange fullSubresourceRange() const = 0;
};

class SwapchainImage : public Image, public SharedObject<SwapchainImage> {
  public:
    SwapchainImage(std::shared_ptr<Device> device, VkImage image, uint32_t width, uint32_t height, VkFormat format);
    ~SwapchainImage();

    uint32_t width() override;
    uint32_t height() override;
    uint32_t depth() override;
    uint32_t layer() override;
    VkFormat &vkFormat() override;
    VkImage &vkImage() override;
    VkImageView &vkImageView(int index = 0) override;
    VkImageLayout &imageLayout() override;
    VkImageSubresourceRange fullSubresourceRange() const override;

  private:
    std::shared_ptr<Device> device_;

    uint32_t width_;
    uint32_t height_;
    uint32_t layer_;
    VkFormat format_;
    VkImageLayout imageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image_ = VK_NULL_HANDLE;
    std::vector<VkImageView> imageViews_{1};
};

class DeviceLocalImage : public Image, public SharedObject<DeviceLocalImage> {
  public:
#ifdef DEBUG
    std::string debugName;
#endif

    DeviceLocalImage(std::shared_ptr<Device> device,
                     std::shared_ptr<VMA> vma,
                     uint32_t width,
                     uint32_t height,
                     uint32_t layer,
                     VkFormat format,
                     VkImageUsageFlags usage
#ifdef DEBUG
                     ,
                     std::string debugName = ""
#endif
    );
    DeviceLocalImage(std::shared_ptr<Device> device,
                     std::shared_ptr<VMA> vma,
                     uint32_t width,
                     uint32_t height,
                     uint32_t depth,
                     uint32_t layer,
                     VkFormat format,
                     VkImageUsageFlags usage
#ifdef DEBUG
                     ,
                     std::string debugName = ""
#endif
    );
    DeviceLocalImage(std::shared_ptr<Device> device,
                     std::shared_ptr<VMA> vma,
                     bool persistStaging,
                     uint32_t width,
                     uint32_t height,
                     uint32_t layer,
                     VkFormat format,
                     VkImageUsageFlags usage
#ifdef DEBUG
                     ,
                     std::string debugName = ""
#endif
    );
    DeviceLocalImage(std::shared_ptr<Device> device,
                     std::shared_ptr<VMA> vma,
                     bool persistStaging,
                     uint32_t width,
                     uint32_t height,
                     uint32_t depth,
                     uint32_t layer,
                     VkFormat format,
                     VkImageUsageFlags usage
#ifdef DEBUG
                     ,
                     std::string debugName = ""
#endif
    );
    DeviceLocalImage(std::shared_ptr<Device> device,
                     std::shared_ptr<VMA> vma,
                     bool persistStaging,
                     uint32_t width,
                     uint32_t height,
                     uint32_t layer,
                     VkFormat format,
                     VkImageUsageFlags usage,
                     VmaAllocationCreateFlags allocationFlags,
                     VmaMemoryUsage vmaUsage,
                     VkImageCreateFlags imageCreateFlags = 0
#ifdef DEBUG
                     ,
                     std::string debugName = ""
#endif
    );
    DeviceLocalImage(std::shared_ptr<Device> device,
                     std::shared_ptr<VMA> vma,
                     bool persistStaging,
                     uint32_t mipLevels,
                     uint32_t width,
                     uint32_t height,
                     uint32_t layer,
                     VkFormat format,
                     VkImageUsageFlags usage,
                     VmaAllocationCreateFlags allocationFlag,
                     VmaMemoryUsage vmaUsage,
                     VkImageCreateFlags imageCreateFlags = 0
#ifdef DEBUG
                     ,
                     std::string debugName = ""
#endif
    );
    DeviceLocalImage(std::shared_ptr<Device> device,
                     std::shared_ptr<VMA> vma,
                     bool persistStaging,
                     uint32_t mipLevels,
                     uint32_t width,
                     uint32_t height,
                     uint32_t depth,
                     uint32_t layer,
                     VkFormat format,
                     VkImageUsageFlags usage,
                     VmaAllocationCreateFlags allocationFlag,
                     VmaMemoryUsage vmaUsage,
                     VkImageCreateFlags imageCreateFlags = 0
#ifdef DEBUG
                     ,
                     std::string debugName = ""
#endif
    );
    // Wraps an image owned by someone else (vanilla). The image is never freed, never transitioned by this class and
    // is assumed to be in `layout`; only the image views created here are destroyed.
    struct ExternalImage {};
    DeviceLocalImage(ExternalImage,
                     std::shared_ptr<Device> device,
                     std::shared_ptr<VMA> vma,
                     VkImage image,
                     uint32_t mipLevels,
                     uint32_t width,
                     uint32_t height,
                     uint32_t layer,
                     VkFormat format,
                     VkImageUsageFlags usage,
                     VkImageLayout layout);
    ~DeviceLocalImage();

    bool isExternal() const;

    void uploadToStagingBuffer(void *src);
    void uploadToImage(VkCommandBuffer cmdBuffer);
    void uploadToImage(std::shared_ptr<CommandBuffer> cmdBuffer);
    void
    uploadToImage(VkCommandBuffer cmdBuffer, std::shared_ptr<Buffer> buffer, std::vector<VkBufferImageCopy> &regions);
    void uploadToImage(std::shared_ptr<CommandBuffer> cmdBuffer,
                       std::shared_ptr<Buffer> buffer,
                       std::vector<VkBufferImageCopy> &regions);

    uint32_t width() override;
    uint32_t height() override;
    uint32_t depth() override;
    uint32_t layer() override;
    VkFormat &vkFormat() override;
    VkBuffer &vkStagingBuffer();
    VkImage &vkImage() override;
    VkImageView &vkImageView(int index = 0) override;
    VkImageLayout &imageLayout() override;
    void *mappedPtr();
    VkImageSubresourceRange fullSubresourceRange() const override;

    void addImageView(VkImageViewCreateInfo info);
    void setMemoryCategory(ResourceMemoryCategory category);

  private:
    static VkImageAspectFlags imageAspectMask(VkImageUsageFlags usage);
    static VkImageSubresourceRange
    makeImageSubresourceRange(VkImageAspectFlags aspectMask, uint32_t mipLevels, uint32_t depth, uint32_t layer);
    static size_t imageByteSize(uint32_t width, uint32_t height, uint32_t depth, uint32_t layer, VkFormat format);

    std::shared_ptr<Device> device_;
    std::shared_ptr<VMA> vma_;

    uint32_t mipLevels_;
    uint32_t width_;
    uint32_t height_;
    uint32_t depth_;
    uint32_t layer_;
    VkFormat format_;
    bool persistStaging_ = false;
    bool external_ = false;
    VkImageUsageFlags usage_;
    VmaAllocationCreateFlags allocationFlags_;
    VmaMemoryUsage vmaUsage_;
    VkImageLayout imageLayout_;
    void *mappedPtr_ = nullptr;
    VkBuffer stagingBuffer_ = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo stagingAllocationInfo_;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo_;
    ResourceMemoryCategory memoryCategory_ = ResourceMemoryCategory::Other;

    std::vector<VkImageView> imageViews_{1};
};

class Sampler : public SharedObject<Sampler> {
  public:
    Sampler(std::shared_ptr<Device> device);
    Sampler(std::shared_ptr<Device> device,
            VkFilter samplingMode,
            VkSamplerMipmapMode mipmapMode,
            VkSamplerAddressMode addressMode);
    Sampler(std::shared_ptr<Device> device,
            VkFilter samplingMode,
            VkSamplerMipmapMode mipmapMode,
            VkSamplerAddressMode addressMode,
            bool compareEnable,
            VkCompareOp compareOp);
    ~Sampler();

    VkSampler vkSamper();
    VkSamplerAddressMode vkAddressMode();
    VkFilter vkSamplingMode();
    VkSamplerMipmapMode vkMipmapMode();

  private:
    std::shared_ptr<Device> device_;

    VkFilter samplingMode_;
    VkSamplerMipmapMode mipmapMode_;
    VkSamplerAddressMode addressMode_;
    bool compareEnable_ = false;
    VkCompareOp compareOp_ = VK_COMPARE_OP_ALWAYS;
    VkSampler samper_;
};

class ImageLoader : public SharedObject<ImageLoader> {
  public:
    ImageLoader(std::vector<std::string> imagePaths, uint32_t forceChannel, bool convertLinearToSrgb = true);
    ~ImageLoader();

    uint32_t width();
    uint32_t height();
    uint32_t channel();
    uint32_t layer();
    void *data();

  private:
    std::vector<std::string> imagePaths_;
    int width_;
    int height_;
    int channel_;
    int layer_;
    std::vector<uint8_t> data_;
};
};
