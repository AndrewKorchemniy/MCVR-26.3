#define VMA_IMPLEMENTATION
#include "core/vulkan/vma.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/instance.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/result.hpp"


vk::VMA::VMA(std::shared_ptr<Instance> instance,
             std::shared_ptr<PhysicalDevice> physicalDevice,
             std::shared_ptr<Device> device) {
    VmaAllocatorCreateInfo allocatorCreateInfo{};
    allocatorCreateInfo.physicalDevice = physicalDevice->vkPhysicalDevice();
    allocatorCreateInfo.device = device->vkDevice();
    allocatorCreateInfo.instance = instance->vkInstance();
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    memoryBudgetEnabled_ = device->hasMemoryBudget();
    if (memoryBudgetEnabled_) { allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT; }
    vkGetPhysicalDeviceMemoryProperties(physicalDevice->vkPhysicalDevice(), &memoryProperties_);

    VkResult importResult = vmaImportVulkanFunctionsFromVolk(&allocatorCreateInfo, &vulkanFunctions_);
    if (importResult != VK_SUCCESS) {
        vk::requireSuccess(importResult, "vmaImportVulkanFunctionsFromVolk");
    }
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions_;

    VkResult createResult = vmaCreateAllocator(&allocatorCreateInfo, &allocator_);
    if (createResult != VK_SUCCESS) {
        vk::requireSuccess(createResult, "vmaCreateAllocator");
    }
}

vk::VMA::~VMA() {
    vmaDestroyAllocator(allocator_);
}

VmaAllocator &vk::VMA::allocator() {
    return allocator_;
}

vk::DeviceLocalMemoryBudget vk::VMA::deviceLocalMemoryBudget() const {
    DeviceLocalMemoryBudget result{};
    if (!memoryBudgetEnabled_ || allocator_ == VK_NULL_HANDLE) { return result; }

    VmaBudget heapBudgets[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(allocator_, heapBudgets);
    for (uint32_t heapIndex = 0; heapIndex < memoryProperties_.memoryHeapCount; heapIndex++) {
        if ((memoryProperties_.memoryHeaps[heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) { continue; }
        result.usage += heapBudgets[heapIndex].usage;
        result.budget += heapBudgets[heapIndex].budget;
    }
    result.valid = result.budget > 0;
    return result;
}
