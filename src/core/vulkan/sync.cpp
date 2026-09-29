#include "core/vulkan/sync.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/result.hpp"

#include <stdexcept>

vk::Semaphore::Semaphore(std::shared_ptr<Device> device) : Semaphore(device, false) {}

vk::Semaphore::Semaphore(std::shared_ptr<Device> device, bool exportable) : device_(device) {
    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
#if defined(_WIN32)
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = exportable ? &exportInfo : nullptr;

    vk::requireSuccess(vkCreateSemaphore(device_->vkDevice(), &semaphoreInfo, nullptr, &semaphore_),
                       "vkCreateSemaphore");
}

vk::Semaphore::~Semaphore() {
    vkDestroySemaphore(device_->vkDevice(), semaphore_, nullptr);
}

VkSemaphore &vk::Semaphore::vkSemaphore() {
    return semaphore_;
}

uint64_t vk::Semaphore::exportHandle() {
#if defined(_WIN32)
    auto getSemaphoreWin32Handle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device_->vkDevice(), "vkGetSemaphoreWin32HandleKHR"));
    if (getSemaphoreWin32Handle == nullptr) {
        throw std::runtime_error("vkGetSemaphoreWin32HandleKHR is unavailable for the selected Vulkan device");
    }

    VkSemaphoreGetWin32HandleInfoKHR getHandleInfo{};
    getHandleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    getHandleInfo.semaphore = semaphore_;
    getHandleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE handle = nullptr;
    vk::requireSuccess(getSemaphoreWin32Handle(device_->vkDevice(), &getHandleInfo, &handle),
                       "vkGetSemaphoreWin32HandleKHR");
    return reinterpret_cast<uint64_t>(handle);
#elif defined(__linux__) || defined(__unix__)
    VkSemaphoreGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    getFdInfo.semaphore = semaphore_;
    getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    vk::requireSuccess(vkGetSemaphoreFdKHR(device_->vkDevice(), &getFdInfo, &fd), "vkGetSemaphoreFdKHR");
    return static_cast<uint64_t>(fd);
#else
    throw std::runtime_error("OpenGL UI external semaphore export is unsupported on this platform");
#endif
}

vk::Fence::Fence(std::shared_ptr<Device> device) : Fence(device, false) {}

vk::Fence::Fence(std::shared_ptr<Device> device, bool signaled) : device_(device) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) { fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; }

    vk::requireSuccess(vkCreateFence(device_->vkDevice(), &fenceInfo, nullptr, &fence_), "vkCreateFence");
}

vk::Fence::~Fence() {
    vkDestroyFence(device_->vkDevice(), fence_, nullptr);
}

VkFence &vk::Fence::vkFence() {
    return fence_;
}
