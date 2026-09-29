#define STB_IMAGE_IMPLEMENTATION
#include "core/vulkan/image.hpp"

#include "core/vulkan/buffer.hpp"
#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/resource_memory_tracker.hpp"
#include "core/vulkan/result.hpp"
#include "core/vulkan/vma.hpp"

#include <cstring>
#include <iostream>

std::ostream &imageCerr() {
    return std::cerr << "[Image] ";
}

VkImageAspectFlags vk::DeviceLocalImage::imageAspectMask(VkImageUsageFlags usage) {
    return (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0 ? VK_IMAGE_ASPECT_DEPTH_BIT :
                                                                        VK_IMAGE_ASPECT_COLOR_BIT;
}

VkImageSubresourceRange vk::DeviceLocalImage::makeImageSubresourceRange(VkImageAspectFlags aspectMask,
                                                                        uint32_t mipLevels,
                                                                        uint32_t depth,
                                                                        uint32_t layer) {
    return {
        .aspectMask = aspectMask,
        .baseMipLevel = 0,
        .levelCount = mipLevels,
        .baseArrayLayer = 0,
        .layerCount = depth > 1 ? 1u : layer,
    };
}

size_t
vk::DeviceLocalImage::imageByteSize(uint32_t width, uint32_t height, uint32_t depth, uint32_t layer, VkFormat format) {
    return static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(depth) *
           static_cast<size_t>(layer) * vk::formatToByte(format);
}

vk::SwapchainImage::SwapchainImage(
    std::shared_ptr<Device> device, VkImage image, uint32_t width, uint32_t height, VkFormat format)
    : device_(device), width_(width), height_(height), layer_(1), format_(format), image_(image) {
    VkImageViewCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = image_;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = format_;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange = wholeColorSubresourceRange;

    if (vkCreateImageView(device_->vkDevice(), &createInfo, nullptr, &imageViews_[0]) != VK_SUCCESS) {
        imageCerr() << "failed to create image view for image" << std::endl;
        exit(EXIT_FAILURE);
    }
}

vk::SwapchainImage::~SwapchainImage() {
    for (auto imageView : imageViews_) {
        if (imageView != VK_NULL_HANDLE) { vkDestroyImageView(device_->vkDevice(), imageView, nullptr); }
    }
}

uint32_t vk::SwapchainImage::width() {
    return width_;
}

uint32_t vk::SwapchainImage::height() {
    return height_;
}

uint32_t vk::SwapchainImage::depth() {
    return 1;
}

uint32_t vk::SwapchainImage::layer() {
    return 1;
}

VkFormat &vk::SwapchainImage::vkFormat() {
    return format_;
}

VkImage &vk::SwapchainImage::vkImage() {
    return image_;
}

VkImageView &vk::SwapchainImage::vkImageView(int index) {
    return imageViews_[index];
}

VkImageLayout &vk::SwapchainImage::imageLayout() {
    return imageLayout_;
}

VkImageSubresourceRange vk::SwapchainImage::fullSubresourceRange() const {
    return wholeColorSubresourceRange;
}

size_t vk::formatToByte(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_USCALED:
        case VK_FORMAT_R8_SSCALED:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8_SRGB: return 1;

        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SNORM:
        case VK_FORMAT_R8G8_USCALED:
        case VK_FORMAT_R8G8_SSCALED:
        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8_SRGB:
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_SNORM:
        case VK_FORMAT_R16_USCALED:
        case VK_FORMAT_R16_SSCALED:
        case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16_SFLOAT: return 2;

        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_R8G8B8_SNORM:
        case VK_FORMAT_R8G8B8_USCALED:
        case VK_FORMAT_R8G8B8_SSCALED:
        case VK_FORMAT_R8G8B8_UINT:
        case VK_FORMAT_R8G8B8_SINT:
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_B8G8R8_UNORM:
        case VK_FORMAT_B8G8R8_SNORM:
        case VK_FORMAT_B8G8R8_USCALED:
        case VK_FORMAT_B8G8R8_SSCALED:
        case VK_FORMAT_B8G8R8_UINT:
        case VK_FORMAT_B8G8R8_SINT:
        case VK_FORMAT_B8G8R8_SRGB: return 3;

        case VK_FORMAT_R16G16B16_UNORM:
        case VK_FORMAT_R16G16B16_SNORM:
        case VK_FORMAT_R16G16B16_USCALED:
        case VK_FORMAT_R16G16B16_SSCALED:
        case VK_FORMAT_R16G16B16_UINT:
        case VK_FORMAT_R16G16B16_SINT:
        case VK_FORMAT_R16G16B16_SFLOAT: return 6;

        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R8G8B8A8_USCALED:
        case VK_FORMAT_R8G8B8A8_SSCALED:
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_UINT:
        case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SNORM:
        case VK_FORMAT_R16G16_USCALED:
        case VK_FORMAT_R16G16_SSCALED:
        case VK_FORMAT_R16G16_UINT:
        case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16_SFLOAT:
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT: return 4;

        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_SNORM:
        case VK_FORMAT_R16G16B16A16_USCALED:
        case VK_FORMAT_R16G16B16A16_SSCALED:
        case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32_SFLOAT: return 8;

        case VK_FORMAT_R32G32B32_UINT:
        case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32_SFLOAT: return 12;

        case VK_FORMAT_R32G32B32A32_UINT:
        case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;

        default: {
            throw std::runtime_error("Format not allowed: " + std::to_string(format));
        }
    }
}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage
#ifdef DEBUG
                                       ,
                                       std::string debugName
#endif
                                       )
    : vk::DeviceLocalImage(device,
                           vma,
                           width,
                           height,
                           1,
                           layer,
                           format,
                           usage
#ifdef DEBUG
                           ,
                           debugName
#endif
      ) {
}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t depth,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage
#ifdef DEBUG
                                       ,
                                       std::string debugName
#endif
                                       )
    : vk::DeviceLocalImage(device,
                           vma,
                           true,
                           width,
                           height,
                           depth,
                           layer,
                           format,
                           usage
#ifdef DEBUG
                           ,
                           debugName
#endif
      ) {
}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       bool persistStaging,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage
#ifdef DEBUG
                                       ,
                                       std::string debugName
#endif
                                       )
    : vk::DeviceLocalImage(device,
                           vma,
                           persistStaging,
                           width,
                           height,
                           1,
                           layer,
                           format,
                           usage
#ifdef DEBUG
                           ,
                           debugName
#endif
      ) {
}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
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
                                       std::string debugName
#endif
                                       )
    : vk::DeviceLocalImage(device,
                           vma,
                           persistStaging,
                           1,
                           width,
                           height,
                           depth,
                           layer,
                           format,
                           usage,
                           0,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                           0
#ifdef DEBUG
                           ,
                           debugName
#endif
      ) {
}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       bool persistStaging,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage,
                                       VmaAllocationCreateFlags allocationFlags,
                                       VmaMemoryUsage vmaUsage,
                                       VkImageCreateFlags imageCreateFlags
#ifdef DEBUG
                                       ,
                                       std::string debugName
#endif
                                       )
    : DeviceLocalImage(device,
                       vma,
                       persistStaging,
                       1,
                       width,
                       height,
                       layer,
                       format,
                       usage,
                       allocationFlags,
                       vmaUsage,
                       imageCreateFlags
#ifdef DEBUG
                       ,
                       debugName
#endif
      ) {
}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       bool persistStaging,
                                       uint32_t mipLevels,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage,
                                       VmaAllocationCreateFlags allocationFlags,
                                       VmaMemoryUsage vmaUsage,
                                       VkImageCreateFlags imageCreateFlags
#ifdef DEBUG
                                       ,
                                       std::string debugName
#endif
                                       )
    : DeviceLocalImage(device,
                       vma,
                       persistStaging,
                       mipLevels,
                       width,
                       height,
                       1,
                       layer,
                       format,
                       usage,
                       allocationFlags,
                       vmaUsage,
                       imageCreateFlags
#ifdef DEBUG
                       ,
                       debugName
#endif
      ) {
}

vk::DeviceLocalImage::DeviceLocalImage(std::shared_ptr<Device> device,
                                       std::shared_ptr<VMA> vma,
                                       bool persistStaging,
                                       uint32_t mipLevels,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t depth,
                                       uint32_t layer,
                                       VkFormat format,
                                       VkImageUsageFlags usage,
                                       VmaAllocationCreateFlags allocationFlags,
                                       VmaMemoryUsage vmaUsage,
                                       VkImageCreateFlags imageCreateFlags
#ifdef DEBUG
                                       ,
                                       std::string debugName
#endif
                                       )
    : device_(device),
      vma_(vma),
      mipLevels_(mipLevels),
      width_(width),
      height_(height),
      depth_(depth),
      layer_(layer),
      format_(format),
      persistStaging_(persistStaging),
      usage_(usage),
      allocationFlags_(allocationFlags),
      vmaUsage_(vmaUsage)
#ifdef DEBUG
      ,
      debugName(debugName)
#endif
{
    if (depth_ == 0) {
        imageCerr() << "image depth must be at least 1" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (depth_ > 1 && layer_ != 1) {
        imageCerr() << "3d images do not support array layers in DeviceLocalImage" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (depth_ > 1 && (imageCreateFlags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0) {
        imageCerr() << "3d images cannot be cube compatible" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (persistStaging_) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = imageByteSize(width_, height_, depth_, layer_, format_);
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        vk::requireSuccess(vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_,
                                           &stagingAllocation_, &stagingAllocationInfo_),
                           "vmaCreateBuffer(image staging buffer)");
        trackVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_);
        mappedPtr_ = stagingAllocationInfo_.pMappedData;
    }

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = imageCreateFlags;
    imageInfo.imageType = depth_ > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    imageInfo.format = format_;
    imageInfo.extent = {width_, height_, depth_};
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = layer_;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | usage_;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    const uint32_t queueFamilies[] = {device_->mainQueueFamilyIndex(), device_->secondaryQueueFamilyIndex()};
    if (queueFamilies[0] != queueFamilies[1] &&
        (usage_ & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) != 0) {
        imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        imageInfo.queueFamilyIndexCount = 2;
        imageInfo.pQueueFamilyIndices = queueFamilies;
    }
    imageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.flags = allocationFlags_;
    allocationInfo.usage = vmaUsage;
    vk::requireSuccess(
        vmaCreateImage(vma_->allocator(), &imageInfo, &allocationInfo, &image_, &allocation_, &allocationInfo_),
        "vmaCreateImage(device-local image)");
    trackVmaAllocation(vma_->allocator(), allocation_, memoryCategory_);

    VkImageViewCreateInfo createInfo = {};
    auto makeDefaultImageViewType = [](uint32_t depth, uint32_t layer) {
        if (depth > 1) { return VK_IMAGE_VIEW_TYPE_3D; }
        return layer == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    };
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = image_;
    createInfo.viewType = makeDefaultImageViewType(depth_, layer_);
    createInfo.format = format_;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange = makeImageSubresourceRange(imageAspectMask(usage_), mipLevels_, depth_, layer_);

    vk::requireSuccess(vkCreateImageView(device_->vkDevice(), &createInfo, nullptr, &imageViews_[0]),
                       "vkCreateImageView(device-local image)");
}

vk::DeviceLocalImage::~DeviceLocalImage() {
    for (int i = 0; i < imageViews_.size(); i++) { vkDestroyImageView(device_->vkDevice(), imageViews_[i], nullptr); }
    untrackVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_);
    untrackVmaAllocation(vma_->allocator(), allocation_, memoryCategory_);
    vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
    vmaDestroyImage(vma_->allocator(), image_, allocation_);
}

void vk::DeviceLocalImage::uploadToStagingBuffer(void *src) {
    if (!persistStaging_) {
        if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE || mappedPtr_ != nullptr) {
            imageCerr() << "if not persist staging, the staging buffer should not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }

        VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = imageByteSize(width_, height_, depth_, layer_, format_);
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        vk::requireSuccess(vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_,
                                           &stagingAllocation_, &stagingAllocationInfo_),
                           "vmaCreateBuffer(transient image staging buffer)");
        trackVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_);
        mappedPtr_ = stagingAllocationInfo_.pMappedData;
    }

    size_t size = imageByteSize(width_, height_, depth_, layer_, format_);
    std::memcpy(mappedPtr_, src, size);
    vmaFlushAllocation(vma_->allocator(), stagingAllocation_, 0, size);

    if (!persistStaging_) {
        untrackVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_);
        vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
        stagingBuffer_ = VK_NULL_HANDLE;
        stagingAllocation_ = VK_NULL_HANDLE;
        mappedPtr_ = nullptr;
    }
}

void vk::DeviceLocalImage::uploadToImage(VkCommandBuffer cmdBuffer) {
    VkBufferImageCopy region = {};
    region.imageSubresource = {usage_ == VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT ?
                                   static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT) :
                                   static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_COLOR_BIT),
                               0, 0, depth_ > 1 ? 1u : layer_};
    region.imageExtent = {width_, height_, depth_};
    vkCmdCopyBufferToImage(cmdBuffer, stagingBuffer_, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void vk::DeviceLocalImage::uploadToImage(std::shared_ptr<CommandBuffer> cmdBuffer) {
    cmdBuffer->retainResource(shared_from_this());
    uploadToImage(cmdBuffer->vkCommandBuffer());
}

void vk::DeviceLocalImage::uploadToImage(VkCommandBuffer cmdBuffer,
                                         std::shared_ptr<Buffer> buffer,
                                         std::vector<VkBufferImageCopy> &regions) {
    vkCmdCopyBufferToImage(cmdBuffer, buffer->vkBuffer(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions.size(),
                           regions.data());
}

void vk::DeviceLocalImage::uploadToImage(std::shared_ptr<CommandBuffer> cmdBuffer,
                                         std::shared_ptr<Buffer> buffer,
                                         std::vector<VkBufferImageCopy> &regions) {
    cmdBuffer->retainResource(shared_from_this());
    cmdBuffer->retainResource(buffer);
    uploadToImage(cmdBuffer->vkCommandBuffer(), buffer, regions);
}

uint32_t vk::DeviceLocalImage::width() {
    return width_;
}

uint32_t vk::DeviceLocalImage::height() {
    return height_;
}

uint32_t vk::DeviceLocalImage::depth() {
    return depth_;
}

uint32_t vk::DeviceLocalImage::layer() {
    return layer_;
}

VkFormat &vk::DeviceLocalImage::vkFormat() {
    return format_;
}

VkBuffer &vk::DeviceLocalImage::vkStagingBuffer() {
    return stagingBuffer_;
}

VkImage &vk::DeviceLocalImage::vkImage() {
    return image_;
}

VkImageView &vk::DeviceLocalImage::vkImageView(int index) {
    return imageViews_[index];
}

VkImageLayout &vk::DeviceLocalImage::imageLayout() {
    return imageLayout_;
}

void *vk::DeviceLocalImage::mappedPtr() {
    return mappedPtr_;
}

VkImageSubresourceRange vk::DeviceLocalImage::fullSubresourceRange() const {
    return makeImageSubresourceRange(imageAspectMask(usage_), mipLevels_, depth_, layer_);
}

void vk::DeviceLocalImage::addImageView(VkImageViewCreateInfo info) {
    VkImageView vkImageView{};
    if (vkCreateImageView(device_->vkDevice(), &info, nullptr, &vkImageView) != VK_SUCCESS) {
        imageCerr() << "failed to create image view for image" << std::endl;
        exit(EXIT_FAILURE);
    }
    imageViews_.push_back(vkImageView);
}

void vk::DeviceLocalImage::setMemoryCategory(ResourceMemoryCategory category) {
    retagVmaAllocation(vma_->allocator(), allocation_, memoryCategory_, category);
    retagVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_, category);
    memoryCategory_ = category;
}

vk::Sampler::Sampler(std::shared_ptr<Device> device)
    : Sampler(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT) {}

vk::Sampler::Sampler(std::shared_ptr<Device> device,
                     VkFilter samplingMode,
                     VkSamplerMipmapMode mipmapMode,
                     VkSamplerAddressMode addressMode)
    : Sampler(device, samplingMode, mipmapMode, addressMode, false, VK_COMPARE_OP_ALWAYS) {}

vk::Sampler::Sampler(std::shared_ptr<Device> device,
                     VkFilter samplingMode,
                     VkSamplerMipmapMode mipmapMode,
                     VkSamplerAddressMode addressMode,
                     bool compareEnable,
                     VkCompareOp compareOp)
    : device_(device),
      samplingMode_(samplingMode),
      mipmapMode_(mipmapMode),
      addressMode_(addressMode),
      compareEnable_(compareEnable),
      compareOp_(compareOp) {
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = samplingMode;
    samplerInfo.minFilter = samplingMode;
    samplerInfo.addressModeU = addressMode;
    samplerInfo.addressModeV = addressMode;
    samplerInfo.addressModeW = addressMode;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 16.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.compareEnable = compareEnable ? VK_TRUE : VK_FALSE;
    samplerInfo.compareOp = compareOp;
    samplerInfo.mipmapMode = mipmapMode;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

    if (vkCreateSampler(device->vkDevice(), &samplerInfo, nullptr, &samper_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler!");
    }
}

vk::Sampler::~Sampler() {
    vkDestroySampler(device_->vkDevice(), samper_, nullptr);
}

VkSampler vk::Sampler::vkSamper() {
    return samper_;
}

VkFilter vk::Sampler::vkSamplingMode() {
    return samplingMode_;
}

VkSamplerMipmapMode vk::Sampler::vkMipmapMode() {
    return mipmapMode_;
}

VkSamplerAddressMode vk::Sampler::vkAddressMode() {
    return addressMode_;
}

std::ostream &imageLoaderCerr() {
    return std::cerr << "[ImageLoader] ";
}

float linearToSrgb(float linear) {
    if (linear <= 0.0031308f) {
        return 12.92f * linear;
    } else {
        return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
    }
}

uint8_t linearToSrgb(uint8_t linear) {
    if (linear / 255.0f <= 0.0031308f) {
        return 12.92f * linear;
    } else {
        return (1.055f * powf(linear / 255.0f, 1.0f / 2.4f) - 0.055f) * 255.0f;
    }
}

vk::ImageLoader::ImageLoader(std::vector<std::string> imagePaths, uint32_t forceChannel, bool convertLinearToSrgb)
    : imagePaths_(imagePaths), channel_(forceChannel), layer_(imagePaths.size()), data_() {
    if (imagePaths.size() == 0) { imageLoaderCerr() << "Cannot load 0 image" << std::endl; }

    for (int i = 0; i < imagePaths_.size(); i++) {
        stbi_uc *imageData;
        int channel;
        if (i == 0) {
            imageData = stbi_load(imagePaths_[i].c_str(), &width_, &height_, &channel, 0);
        } else {
            int currentWidth, currentHeight;
            imageData = stbi_load(imagePaths_[i].c_str(), &currentWidth, &currentHeight, &channel, 0);
            if (currentWidth != width_ || currentHeight != height_) {
                imageLoaderCerr() << "images are not with the same shape" << std::endl;
                imageLoaderCerr() << "current: [width=" << currentWidth << ", height=" << currentHeight << "]"
                                  << std::endl;
                imageLoaderCerr() << "existing: [width=" << width_ << ", height=" << height_ << "]" << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        if (forceChannel < channel) {
            imageLoaderCerr() << "Cannot compress image" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (int h = 0; h < height_; h++) {
            for (int w = 0; w < width_; w++) {
                const int srcIndex = (h * width_ + w) * channel;
                for (int c = 0; c < std::min(channel, 3); c++) {
                    uint8_t value = imageData[srcIndex + c];
                    data_.push_back(convertLinearToSrgb ? linearToSrgb(value) : value);
                }

                if (channel == 3 && forceChannel == 4) {
                    data_.push_back(255);
                } else if (channel == 1 && forceChannel == 4) {
                    uint8_t value = imageData[srcIndex];
                    value = convertLinearToSrgb ? linearToSrgb(value) : value;
                    data_.push_back(value);
                    data_.push_back(value);
                    data_.push_back(255);
                } else if (channel == 4 && forceChannel == 4) {
                    data_.push_back(imageData[srcIndex + 3]);
                } else {
                    imageLoaderCerr() << "Force channel of " << forceChannel << " is not support for channel "
                                      << channel << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }

        stbi_image_free(imageData);
    }
}

vk::ImageLoader::~ImageLoader() {}

uint32_t vk::ImageLoader::width() {
    return width_;
}

uint32_t vk::ImageLoader::height() {
    return height_;
}

uint32_t vk::ImageLoader::channel() {
    return channel_;
}

uint32_t vk::ImageLoader::layer() {
    return layer_;
}

void *vk::ImageLoader::data() {
    return data_.data();
}
