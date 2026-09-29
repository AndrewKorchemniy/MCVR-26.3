#pragma once

#include "core/all_extern.hpp"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace vk {
class Instance;
class Window;

using DeviceUuid = std::array<uint8_t, VK_UUID_SIZE>;

struct OpenGLDeviceIdentity {
    std::vector<DeviceUuid> deviceUuids;
    DeviceUuid driverUuid{};
};

class PhysicalDevice : public SharedObject<PhysicalDevice> {
  public:
    PhysicalDevice(std::shared_ptr<Instance> instance, std::shared_ptr<Window> window);
    PhysicalDevice(std::shared_ptr<Instance> instance,
                   std::shared_ptr<Window> window,
                   OpenGLDeviceIdentity openGLDeviceIdentity);
    ~PhysicalDevice();

    VkPhysicalDevice &vkPhysicalDevice();
    uint32_t mainQueueIndex();
    uint32_t secondaryQueueIndex();
    uint32_t secondaryQueueOffset();
    bool hasIndependentSecondaryQueue() const;

    void findPhysicalDevice();
    void findQueueFamilies();

    VkPhysicalDeviceProperties properties();
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties();
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructProperties();

  private:
    static constexpr uint32_t invalidQueueFamily = std::numeric_limits<uint32_t>::max();

    struct QueueFamilySelection {
        uint32_t mainIndex = invalidQueueFamily;
        uint32_t secondaryIndex = invalidQueueFamily;
        uint32_t secondaryOffset = 0;
        bool independentSecondaryQueue = false;
    };

    struct PhysicalDeviceCandidate {
        VkPhysicalDevice device = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties properties{};
        QueueFamilySelection queueFamilies{};
        DeviceUuid deviceUuid{};
        DeviceUuid driverUuid{};
        bool matchesOpenGLDevice = false;
        bool matchesOpenGLDriver = false;
        std::vector<std::string> rejectionReasons;
    };

    static uint32_t physicalDevicePreference(VkPhysicalDeviceType type);
    static std::string resultDescription(VkResult result);
    static bool findRequiredQueueFamilies(VkPhysicalDevice device,
                                          VkSurfaceKHR surface,
                                          QueueFamilySelection &selection,
                                          std::string &rejectionReason);
    static void evaluateRequiredExtensions(PhysicalDeviceCandidate &candidate);
    static void evaluateRequiredFeatures(PhysicalDeviceCandidate &candidate);
    static void evaluatePresentationSupport(PhysicalDeviceCandidate &candidate, VkSurfaceKHR surface);
    static void evaluateOpenGLDeviceIdentity(PhysicalDeviceCandidate &candidate,
                                             const OpenGLDeviceIdentity &openGLDeviceIdentity);
    static PhysicalDeviceCandidate evaluatePhysicalDevice(
        VkPhysicalDevice device, VkSurfaceKHR surface, const std::optional<OpenGLDeviceIdentity> &openGLDeviceIdentity);

    PhysicalDevice(std::shared_ptr<Instance> instance,
                   std::shared_ptr<Window> window,
                   std::optional<OpenGLDeviceIdentity> openGLDeviceIdentity);

    std::shared_ptr<Instance> instance_;
    std::shared_ptr<Window> window_;
    std::optional<OpenGLDeviceIdentity> openGLDeviceIdentity_;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    uint32_t mainQueueIndex_ = invalidQueueFamily;
    uint32_t secondaryQueueIndex_ = invalidQueueFamily;
    uint32_t secondaryQueueOffset_ = 0;
    bool independentSecondaryQueue_ = false;

    VkPhysicalDeviceProperties properties_;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties_;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructProperties_;
};
}
