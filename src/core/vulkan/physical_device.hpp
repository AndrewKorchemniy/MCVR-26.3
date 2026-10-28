#pragma once

#include "core/all_extern.hpp"

#include <cstdint>

namespace vk {
class Instance;

// Queues vanilla created. MCVR may use the graphics queue only from the render thread and the compute queue freely,
// since vanilla never touches it. The transfer queue is recorded but not used.
struct AdoptedQueues {
    uint32_t graphicsFamily = 0;
    VkQueue graphics = VK_NULL_HANDLE;
    uint32_t computeFamily = 0;
    VkQueue compute = VK_NULL_HANDLE;
    uint32_t transferFamily = 0;
    VkQueue transfer = VK_NULL_HANDLE;
};

// Non-owning wrapper around the VkPhysicalDevice vanilla picked.
class PhysicalDevice : public SharedObject<PhysicalDevice> {
  public:
    PhysicalDevice(std::shared_ptr<Instance> instance, VkPhysicalDevice physicalDevice, AdoptedQueues queues);
    ~PhysicalDevice();

    VkPhysicalDevice &vkPhysicalDevice();
    uint32_t mainQueueIndex();
    uint32_t secondaryQueueIndex();
    bool hasIndependentSecondaryQueue() const;
    const AdoptedQueues &queues() const;

    VkPhysicalDeviceProperties properties();
    VkPhysicalDeviceFeatures supportedFeatures() const;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties();
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructProperties();

  private:
    std::shared_ptr<Instance> instance_;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    AdoptedQueues queues_;
    uint32_t mainQueueIndex_ = 0;
    uint32_t secondaryQueueIndex_ = 0;
    bool independentSecondaryQueue_ = false;

    VkPhysicalDeviceProperties properties_;
    VkPhysicalDeviceFeatures supportedFeatures_{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties_;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructProperties_;
};
}
