#pragma once

#include "core/all_extern.hpp"

#include <cstdint>
#include <vector>

namespace vk {
class DeviceLocalBuffer;
class CommandBuffer;
class PhysicalDevice;
class Device;
class VMA;
class RayTracingPipeline;

class SBT : public SharedObject<SBT> {
  public:
    struct HitGroupRecordRangeInfo {
        uint32_t offset = 0;
        uint32_t count = 0;
    };

    SBT(std::shared_ptr<PhysicalDevice> physicalDevice,
        std::shared_ptr<Device> device,
        std::shared_ptr<VMA> vma,
        std::shared_ptr<RayTracingPipeline> pipeline,
        uint32_t missCount,
        uint32_t hitCount);
    ~SBT();

    void uploadStaticSBT(std::shared_ptr<CommandBuffer> commandBuffer);
    void setupHitSBT(const std::vector<uint32_t> &prefixHitGroupIndices,
                     const std::vector<uint32_t> &cachedSuffixHitGroupIndices,
                     const std::vector<HitGroupRecordRangeInfo> &cachedSuffixDirtyRanges,
                     uint64_t cachedSuffixVersion,
                     std::shared_ptr<CommandBuffer> commandBuffer);

    VkStridedDeviceAddressRegionKHR &raygenRegion();
    VkStridedDeviceAddressRegionKHR &missRegion();
    VkStridedDeviceAddressRegionKHR &hitRegion();
    VkStridedDeviceAddressRegionKHR &callableRegion();

  private:
    std::shared_ptr<vk::VMA> vma_;
    std::shared_ptr<vk::Device> device_;

    uint32_t handleSize_;
    uint32_t alignedHandleSize_;
    uint32_t baseAlignment_;
    uint32_t missCount_;

    std::vector<uint8_t> shaderHandleStorage_;
    std::vector<uint8_t> hitGroupRecords_;
    std::vector<uint32_t> cachedPrefixHitGroupIndices_;
    uint32_t cachedSuffixOffset_ = 0;
    uint32_t cachedSuffixCount_ = 0;
    uint64_t cachedSuffixVersion_ = 0;
    bool hasValidCachedPrefix_ = false;
    bool hasValidCachedSuffix_ = false;

    VkStridedDeviceAddressRegionKHR raygenRegion_{};
    VkStridedDeviceAddressRegionKHR missRegion_{};
    VkStridedDeviceAddressRegionKHR hitRegion_{};
    VkStridedDeviceAddressRegionKHR callableRegion_{};

    std::shared_ptr<DeviceLocalBuffer> rgenSBT_;
    std::shared_ptr<DeviceLocalBuffer> rmissSBT_;
    std::shared_ptr<DeviceLocalBuffer> rhitSBT_;

    void writeHitGroupRecord(uint8_t *dst, uint32_t hitGroupIndex) const;
};
};
