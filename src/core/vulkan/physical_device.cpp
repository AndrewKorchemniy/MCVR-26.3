#include "core/vulkan/physical_device.hpp"

#include "core/vulkan/instance.hpp"

#include <stdexcept>
#include <utility>

vk::PhysicalDevice::PhysicalDevice(std::shared_ptr<Instance> instance,
                                   VkPhysicalDevice physicalDevice,
                                   AdoptedQueues queues)
    : instance_(std::move(instance)), physicalDevice_(physicalDevice), queues_(queues) {
    if (physicalDevice_ == VK_NULL_HANDLE) { throw std::invalid_argument("Vulkan physical device handle is null"); }
    if (queues_.graphics == VK_NULL_HANDLE) { throw std::invalid_argument("Vulkan graphics queue handle is null"); }

    mainQueueIndex_ = queues_.graphicsFamily;
    // A compute queue that is the graphics queue itself gives no asynchronous capability, so it is not treated as
    // a separate queue.
    independentSecondaryQueue_ = queues_.compute != VK_NULL_HANDLE && queues_.compute != queues_.graphics;
    secondaryQueueIndex_ = independentSecondaryQueue_ ? queues_.computeFamily : queues_.graphicsFamily;

    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelStructProperties{};
    accelStructProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties{};
    rayTracingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    rayTracingProperties.pNext = &accelStructProperties;

    VkPhysicalDeviceProperties2 deviceProps2{};
    deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProps2.pNext = &rayTracingProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &deviceProps2);

    properties_ = deviceProps2.properties;
    rayTracingProperties_ = rayTracingProperties;
    rayTracingProperties_.pNext = nullptr;
    accelerationStructProperties_ = accelStructProperties;

    vkGetPhysicalDeviceFeatures(physicalDevice_, &supportedFeatures_);
}

vk::PhysicalDevice::~PhysicalDevice() {
}

VkPhysicalDevice &vk::PhysicalDevice::vkPhysicalDevice() {
    return physicalDevice_;
}

uint32_t vk::PhysicalDevice::mainQueueIndex() {
    return mainQueueIndex_;
}

uint32_t vk::PhysicalDevice::secondaryQueueIndex() {
    return secondaryQueueIndex_;
}

bool vk::PhysicalDevice::hasIndependentSecondaryQueue() const {
    return independentSecondaryQueue_;
}

const vk::AdoptedQueues &vk::PhysicalDevice::queues() const {
    return queues_;
}

VkPhysicalDeviceProperties vk::PhysicalDevice::properties() {
    return properties_;
}

VkPhysicalDeviceFeatures vk::PhysicalDevice::supportedFeatures() const {
    return supportedFeatures_;
}

VkPhysicalDeviceRayTracingPipelinePropertiesKHR vk::PhysicalDevice::rayTracingProperties() {
    return rayTracingProperties_;
}

VkPhysicalDeviceAccelerationStructurePropertiesKHR vk::PhysicalDevice::accelerationStructProperties() {
    return accelerationStructProperties_;
}
