#pragma once

#include "core/all_extern.hpp"

#include <mutex>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace vk {
class Instance;
class PhysicalDevice;

// Non-owning wrapper around the VkDevice vanilla created with Radiance's features enabled. The queues are vanilla's;
// the graphics queue may only be used from the render thread.
class Device : public SharedObject<Device> {
  public:
    Device(std::shared_ptr<Instance> instance,
           std::shared_ptr<PhysicalDevice> physicalDevice,
           VkDevice device,
           const std::vector<std::string> &enabledExtensions);
    ~Device();

    VkDevice &vkDevice();
    VkPipelineCache vkPipelineCache() const;
    void savePipelineCache();
    VkQueue &mainVkQueue();
    VkQueue &presentQueue();
    VkQueue &secondaryQueue();
    VkQueue &asyncComputeQueue();
    uint32_t mainQueueFamilyIndex() const;
    uint32_t secondaryQueueFamilyIndex() const;
    bool hasIndependentSecondaryQueue() const;
    std::recursive_mutex &mainQueueMutex();
    std::recursive_mutex &presentQueueMutex();
    std::recursive_mutex &secondaryQueueMutex();
    std::recursive_mutex &asyncComputeQueueMutex();

    bool hasEnabledExtension(const char *name) const;
    bool hasExtendedDynamicState2LogicOp() const;
    bool hasMultiDrawIndirect() const;
    bool hasDrawIndirectFirstInstance() const;
    bool hasShaderInvocationReorderEXT() const;
    bool hasShaderInvocationReorderNV() const;
    bool isDlssRRDeviceExtensionsCompatible() const;
    bool isDlssSRDeviceExtensionsCompatible() const;
    bool isDlssFrameGenerationDeviceExtensionsCompatible() const;
    bool isFsrFrameGenerationDeviceExtensionsCompatible() const;
    bool isXessDeviceExtensionsCompatible() const;
    bool hasMemoryBudget() const;

  private:
    std::shared_ptr<Instance> instance_;
    std::shared_ptr<PhysicalDevice> physicalDevice_;

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::filesystem::path pipelineCachePath_;
    void initPipelineCache();
    std::unordered_set<std::string> enabledExtensions_;
    VkQueue mainQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkQueue secondaryQueue_ = VK_NULL_HANDLE;
    VkQueue asyncComputeQueue_ = VK_NULL_HANDLE;
    std::recursive_mutex mainQueueMutex_;
    std::recursive_mutex presentQueueMutex_;
    std::recursive_mutex secondaryQueueMutex_;
    std::recursive_mutex asyncComputeQueueMutex_;

    bool extendedDynamicState2LogicOp_ = false;
    bool multiDrawIndirect_ = false;
    bool drawIndirectFirstInstance_ = false;
    bool shaderInvocationReorderEXT_ = false;
    bool shaderInvocationReorderNV_ = false;
    bool dlssRRDeviceExtensionsCompatible_ = false;
    bool dlssSRDeviceExtensionsCompatible_ = false;
    bool dlssFrameGenerationDeviceExtensionsCompatible_ = false;
    bool fsrFrameGenerationDeviceExtensionsCompatible_ = false;
    bool xessDeviceExtensionsCompatible_ = false;
    bool memoryBudgetEnabled_ = false;
    bool independentSecondaryQueue_ = false;
};
};
