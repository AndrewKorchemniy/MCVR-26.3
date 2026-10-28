#include "core/vulkan/sync.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/result.hpp"

#include <stdexcept>

vk::Semaphore::Semaphore(std::shared_ptr<Device> device) : device_(device) {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    vk::requireSuccess(vkCreateSemaphore(device_->vkDevice(), &semaphoreInfo, nullptr, &semaphore_),
                       "vkCreateSemaphore");
}

vk::Semaphore::~Semaphore() {
    vkDestroySemaphore(device_->vkDevice(), semaphore_, nullptr);
}

VkSemaphore &vk::Semaphore::vkSemaphore() {
    return semaphore_;
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
