#pragma once

#include <deque>
#include <functional>

#include "core/all_extern.hpp"

namespace vk {
class PhysicalDevice;
class Device;
class Instance;

struct DeviceLocalMemoryBudget {
    VkDeviceSize usage = 0;
    VkDeviceSize budget = 0;
    bool valid = false;

    float pressure() const {
        return valid && budget > 0 ? static_cast<float>(static_cast<double>(usage) / static_cast<double>(budget)) :
                                     0.0f;
    }
};

class VMA : public SharedObject<VMA> {
  public:
    VMA(std::shared_ptr<Instance> instance,
        std::shared_ptr<PhysicalDevice> physicalDevice,
        std::shared_ptr<Device> device);
    ~VMA();

    VmaAllocator &allocator();
    DeviceLocalMemoryBudget deviceLocalMemoryBudget() const;

  private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VmaVulkanFunctions vulkanFunctions_{};
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    bool memoryBudgetEnabled_ = false;
};
};
