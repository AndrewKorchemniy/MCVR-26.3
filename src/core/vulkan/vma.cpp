#define VMA_IMPLEMENTATION
#include "core/vulkan/vma.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/instance.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/result.hpp"

#include <cstdio>

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

    // vmaImportVulkanFunctionsFromVolk reloads a whole device table through vkGetDeviceProcAddr, which crashes inside
    // NVIDIA's driver on Minecraft's device. volk's global pointers were already loaded for it by volkLoadDevice.
    vulkanFunctions_ = {};
    vulkanFunctions_.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions_.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    vulkanFunctions_.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    vulkanFunctions_.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    vulkanFunctions_.vkAllocateMemory = vkAllocateMemory;
    vulkanFunctions_.vkFreeMemory = vkFreeMemory;
    vulkanFunctions_.vkMapMemory = vkMapMemory;
    vulkanFunctions_.vkUnmapMemory = vkUnmapMemory;
    vulkanFunctions_.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    vulkanFunctions_.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
    vulkanFunctions_.vkBindBufferMemory = vkBindBufferMemory;
    vulkanFunctions_.vkBindImageMemory = vkBindImageMemory;
    vulkanFunctions_.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
    vulkanFunctions_.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    vulkanFunctions_.vkCreateBuffer = vkCreateBuffer;
    vulkanFunctions_.vkDestroyBuffer = vkDestroyBuffer;
    vulkanFunctions_.vkCreateImage = vkCreateImage;
    vulkanFunctions_.vkDestroyImage = vkDestroyImage;
    vulkanFunctions_.vkCmdCopyBuffer = vkCmdCopyBuffer;
    // Minecraft enables neither VK_KHR_get_physical_device_properties2, VK_KHR_get_memory_requirements2,
    // VK_KHR_bind_memory2 nor VK_KHR_maintenance4-as-KHR-names, so volk resolves those extension entry points to null.
    // The device is Vulkan 1.3, where they are core under their unsuffixed names.
    if (vulkanFunctions_.vkGetBufferMemoryRequirements2KHR == nullptr) {
        vulkanFunctions_.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2;
    }
    if (vulkanFunctions_.vkGetImageMemoryRequirements2KHR == nullptr) {
        vulkanFunctions_.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2;
    }
    if (vulkanFunctions_.vkBindBufferMemory2KHR == nullptr) { vulkanFunctions_.vkBindBufferMemory2KHR = vkBindBufferMemory2; }
    if (vulkanFunctions_.vkBindImageMemory2KHR == nullptr) { vulkanFunctions_.vkBindImageMemory2KHR = vkBindImageMemory2; }
    if (vulkanFunctions_.vkGetPhysicalDeviceMemoryProperties2KHR == nullptr) {
        vulkanFunctions_.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;
    }
    if (vulkanFunctions_.vkGetDeviceBufferMemoryRequirements == nullptr) {
        vulkanFunctions_.vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements;
    }
    if (vulkanFunctions_.vkGetDeviceImageMemoryRequirements == nullptr) {
        vulkanFunctions_.vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirements;
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
