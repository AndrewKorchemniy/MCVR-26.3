#include "core/render/opengl_ui/opengl_ui_surface.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/result.hpp"

#include <stdexcept>

extern "C" const char MCVR_OPENGL_UI_BACKEND_MARKER[] = "MCVR_UI_BACKEND_OPENGL OpenGLUiSurface";

namespace mcvr {

#if defined(_WIN32)
constexpr VkExternalMemoryHandleTypeFlagBits externalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr VkExternalMemoryHandleTypeFlagBits externalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

OpenGLUiSurface::OpenGLUiSurface(std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                                 std::shared_ptr<vk::Device> device,
                                 uint32_t generation,
                                 uint32_t frameIndex,
                                 uint32_t width,
                                 uint32_t height)
    : physicalDevice_(physicalDevice),
      device_(device),
      generation_(generation),
      frameIndex_(frameIndex),
      width_(width),
      height_(height) {
    verifyExternalImageSupport();

    VkExternalMemoryImageCreateInfo externalImageInfo{};
    externalImageInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalImageInfo.handleTypes = externalMemoryHandleType;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalImageInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format_;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vk::requireSuccess(vkCreateImage(device_->vkDevice(), &imageInfo, nullptr, &image_),
                       "vkCreateImage(OpenGL UI external image)");

    VkMemoryDedicatedRequirements dedicatedRequirements{};
    dedicatedRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;

    VkMemoryRequirements2 memoryRequirements{};
    memoryRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    memoryRequirements.pNext = &dedicatedRequirements;

    VkImageMemoryRequirementsInfo2 memoryRequirementsInfo{};
    memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
    memoryRequirementsInfo.image = image_;
    vkGetImageMemoryRequirements2(device_->vkDevice(), &memoryRequirementsInfo, &memoryRequirements);
    memorySize_ = memoryRequirements.memoryRequirements.size;

    VkExportMemoryAllocateInfo exportAllocateInfo{};
    exportAllocateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocateInfo.handleTypes = externalMemoryHandleType;

    VkMemoryDedicatedAllocateInfo dedicatedAllocateInfo{};
    dedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedAllocateInfo.image = image_;
    exportAllocateInfo.pNext = &dedicatedAllocateInfo;

    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext = &exportAllocateInfo;
    allocateInfo.allocationSize = memoryRequirements.memoryRequirements.size;
    allocateInfo.memoryTypeIndex =
        findMemoryType(memoryRequirements.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vk::requireSuccess(vkAllocateMemory(device_->vkDevice(), &allocateInfo, nullptr, &memory_),
                       "vkAllocateMemory(OpenGL UI external image)");
    vk::requireSuccess(vkBindImageMemory(device_->vkDevice(), image_, memory_, 0),
                       "vkBindImageMemory(OpenGL UI external image)");

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format_;
    viewInfo.components = {
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
    };
    viewInfo.subresourceRange = vk::wholeColorSubresourceRange;
    vk::requireSuccess(vkCreateImageView(device_->vkDevice(), &viewInfo, nullptr, &imageView_),
                       "vkCreateImageView(OpenGL UI external image)");

    vulkanReadySemaphore_ = vk::Semaphore::create(device_, true);
    glCompleteSemaphore_ = vk::Semaphore::create(device_, true);
}

OpenGLUiSurface::~OpenGLUiSurface() {
    if (imageView_ != VK_NULL_HANDLE) { vkDestroyImageView(device_->vkDevice(), imageView_, nullptr); }
    if (image_ != VK_NULL_HANDLE) { vkDestroyImage(device_->vkDevice(), image_, nullptr); }
    if (memory_ != VK_NULL_HANDLE) { vkFreeMemory(device_->vkDevice(), memory_, nullptr); }
}

uint32_t OpenGLUiSurface::generation() const {
    return generation_;
}

uint32_t OpenGLUiSurface::frameIndex() const {
    return frameIndex_;
}

uint64_t OpenGLUiSurface::memorySize() const {
    return memorySize_;
}

VkImageLayout &OpenGLUiSurface::imageLayout() {
    return imageLayout_;
}

VkImageSubresourceRange OpenGLUiSurface::fullSubresourceRange() const {
    return vk::wholeColorSubresourceRange;
}

VkSemaphore OpenGLUiSurface::vulkanReadySemaphore() const {
    return vulkanReadySemaphore_->vkSemaphore();
}

VkSemaphore OpenGLUiSurface::glCompleteSemaphore() const {
    return glCompleteSemaphore_->vkSemaphore();
}

OpenGLUiSurfaceExport OpenGLUiSurface::exportHandles() {
#if defined(_WIN32)
    auto getMemoryWin32Handle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(device_->vkDevice(), "vkGetMemoryWin32HandleKHR"));
    if (getMemoryWin32Handle == nullptr) {
        throw std::runtime_error("vkGetMemoryWin32HandleKHR is unavailable for the selected Vulkan device");
    }

    VkMemoryGetWin32HandleInfoKHR getHandleInfo{};
    getHandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    getHandleInfo.memory = memory_;
    getHandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE memoryHandle = nullptr;
    vk::requireSuccess(getMemoryWin32Handle(device_->vkDevice(), &getHandleInfo, &memoryHandle),
                       "vkGetMemoryWin32HandleKHR(OpenGL UI external image)");

    return {
        .generation = generation_,
        .frameIndex = frameIndex_,
        .width = width_,
        .height = height_,
        .memorySize = memorySize_,
        .memoryHandle = reinterpret_cast<uint64_t>(memoryHandle),
        .vulkanReadySemaphoreHandle = vulkanReadySemaphore_->exportHandle(),
        .glCompleteSemaphoreHandle = glCompleteSemaphore_->exportHandle(),
    };
#elif defined(__linux__) || defined(__unix__)
    VkMemoryGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = memory_;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int memoryFd = -1;
    vk::requireSuccess(vkGetMemoryFdKHR(device_->vkDevice(), &getFdInfo, &memoryFd),
                       "vkGetMemoryFdKHR(OpenGL UI external image)");

    return {
        .generation = generation_,
        .frameIndex = frameIndex_,
        .width = width_,
        .height = height_,
        .memorySize = memorySize_,
        .memoryHandle = static_cast<uint64_t>(memoryFd),
        .vulkanReadySemaphoreHandle = vulkanReadySemaphore_->exportHandle(),
        .glCompleteSemaphoreHandle = glCompleteSemaphore_->exportHandle(),
    };
#else
    throw std::runtime_error("OpenGL UI external image export is unsupported on this platform");
#endif
}

bool OpenGLUiSurface::ownedByExternal() const {
    return ownedByExternal_;
}

void OpenGLUiSurface::setOwnedByExternal(bool ownedByExternal) {
    ownedByExternal_ = ownedByExternal;
}

uint32_t OpenGLUiSurface::width() {
    return width_;
}

uint32_t OpenGLUiSurface::height() {
    return height_;
}

uint32_t OpenGLUiSurface::depth() {
    return depth_;
}

uint32_t OpenGLUiSurface::layer() {
    return layer_;
}

VkFormat &OpenGLUiSurface::vkFormat() {
    return format_;
}

VkImage &OpenGLUiSurface::vkImage() {
    return image_;
}

VkImageView &OpenGLUiSurface::vkImageView(int) {
    return imageView_;
}

uint32_t OpenGLUiSurface::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_->vkPhysicalDevice(), &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) == 0) { continue; }
        if ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) { return i; }
    }

    throw std::runtime_error("No compatible memory type for OpenGL UI external image");
}

void OpenGLUiSurface::verifyExternalImageSupport() const {
    VkPhysicalDeviceExternalImageFormatInfo externalInfo{};
    externalInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
    externalInfo.handleType = externalMemoryHandleType;

    VkPhysicalDeviceImageFormatInfo2 formatInfo{};
    formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    formatInfo.pNext = &externalInfo;
    formatInfo.format = format_;
    formatInfo.type = VK_IMAGE_TYPE_2D;
    formatInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    formatInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    formatInfo.flags = 0;

    VkExternalImageFormatProperties externalProperties{};
    externalProperties.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;

    VkImageFormatProperties2 imageFormatProperties{};
    imageFormatProperties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    imageFormatProperties.pNext = &externalProperties;

    VkResult result = vkGetPhysicalDeviceImageFormatProperties2(physicalDevice_->vkPhysicalDevice(), &formatInfo,
                                                                &imageFormatProperties);
    if (result != VK_SUCCESS) {
        vk::requireSuccess(result, "vkGetPhysicalDeviceImageFormatProperties2(OpenGL UI external image)");
    }

    VkExternalMemoryFeatureFlags features = externalProperties.externalMemoryProperties.externalMemoryFeatures;
    if ((features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) == 0 ||
        (features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0) {
        throw std::runtime_error("OpenGL UI external image format is not exportable/importable");
    }
}

}
