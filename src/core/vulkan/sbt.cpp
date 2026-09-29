#include "core/vulkan/sbt.hpp"

#include "core/profiler/profiler.hpp"
#include "core/vulkan/buffer.hpp"
#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/pipeline.hpp"
#include "core/vulkan/vma.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

static uint32_t align(uint32_t addr, uint32_t alignment) {
    return (addr + alignment - 1) & ~(alignment - 1);
}

vk::SBT::SBT(std::shared_ptr<PhysicalDevice> physicalDevice,
             std::shared_ptr<Device> device,
             std::shared_ptr<VMA> vma,
             std::shared_ptr<RayTracingPipeline> pipeline,
             uint32_t missCount,
             uint32_t hitCount)
    : vma_(vma), device_(device), missCount_(missCount) {
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties = physicalDevice->rayTracingProperties();

    handleSize_ = rayTracingProperties.shaderGroupHandleSize;
    uint32_t handleAlignment = rayTracingProperties.shaderGroupHandleAlignment;
    baseAlignment_ = rayTracingProperties.shaderGroupBaseAlignment;
    uint32_t groupCount = 1 + missCount + hitCount;

    alignedHandleSize_ = handleSize_;
    if (handleAlignment > 0) { alignedHandleSize_ = align(handleSize_, handleAlignment); }

    shaderHandleStorage_.resize(groupCount * handleSize_);
    vkGetRayTracingShaderGroupHandlesKHR(device->vkDevice(), pipeline->vkPipeline(), 0, groupCount,
                                         shaderHandleStorage_.size(), shaderHandleStorage_.data());

    rgenSBT_ = DeviceLocalBuffer::create(vma, device, false, handleSize_,
                                         VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, baseAlignment_);
    rmissSBT_ = DeviceLocalBuffer::create(vma, device, false, alignedHandleSize_ * missCount,
                                          VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                          0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, baseAlignment_);

    rgenSBT_->uploadToStagingBuffer(&shaderHandleStorage_[0]);
    rmissSBT_->uploadToStagingBuffer(&shaderHandleStorage_[handleSize_]);

    raygenRegion_.deviceAddress = rgenSBT_->bufferAddress();
    raygenRegion_.stride = alignedHandleSize_;
    raygenRegion_.size = alignedHandleSize_;

    missRegion_.deviceAddress = rmissSBT_->bufferAddress();
    missRegion_.stride = alignedHandleSize_;
    missRegion_.size = alignedHandleSize_ * missCount_;
}

vk::SBT::~SBT() {}

void vk::SBT::uploadStaticSBT(std::shared_ptr<CommandBuffer> commandBuffer) {
    auto addBarrier = [commandBuffer](std::shared_ptr<vk::DeviceLocalBuffer> buffer) {
        commandBuffer->barriersBufferImage({vk::CommandBuffer::BufferMemoryBarrier{
                                               .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                               .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
                                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                               .buffer = buffer,
                                           }},
                                           {});
    };

    rgenSBT_->uploadToBuffer(commandBuffer);
    addBarrier(rgenSBT_);
    rmissSBT_->uploadToBuffer(commandBuffer);
    addBarrier(rmissSBT_);
}

void vk::SBT::writeHitGroupRecord(uint8_t *dst, uint32_t hitGroupIndex) const {
    const uint32_t paletteOffset = (1 + missCount_ + hitGroupIndex) * handleSize_;
    const uint8_t *src = shaderHandleStorage_.data() + paletteOffset;
    std::memcpy(dst, src, handleSize_);
}

void vk::SBT::setupHitSBT(const std::vector<uint32_t> &prefixHitGroupIndices,
                          const std::vector<uint32_t> &cachedSuffixHitGroupIndices,
                          const std::vector<HitGroupRecordRangeInfo> &cachedSuffixDirtyRanges,
                          uint64_t cachedSuffixVersion,
                          std::shared_ptr<CommandBuffer> commandBuffer) {
    MCVR_PROFILE_SCOPE("native.SBT.setupHitSBT");
    const uint32_t prefixHitGroupCount = static_cast<uint32_t>(prefixHitGroupIndices.size());
    const uint32_t cachedSuffixCount = static_cast<uint32_t>(cachedSuffixHitGroupIndices.size());
    const uint32_t hitGroupCount = prefixHitGroupCount + cachedSuffixCount;
    VkDeviceSize rhitSBTSize = static_cast<VkDeviceSize>(hitGroupCount) * alignedHandleSize_;
    MCVR_PROFILE_METRIC("sbt.prefix_hit_group_count", prefixHitGroupCount);
    MCVR_PROFILE_METRIC("sbt.cached_suffix_hit_group_count", cachedSuffixCount);
    MCVR_PROFILE_METRIC("sbt.hit_group_count", hitGroupCount);
    MCVR_PROFILE_METRIC("sbt.hit_sbt_bytes", rhitSBTSize);
    MCVR_PROFILE_METRIC("sbt.cached_suffix_dirty_range_count", cachedSuffixDirtyRanges.size());
    if (rhitSBTSize == 0) {
        std::cerr << "Hit group should contains something!" << std::endl;
        exit(1);
    }

    const bool shouldResizeRecords = hitGroupRecords_.size() != rhitSBTSize;
    if (shouldResizeRecords) {
        MCVR_PROFILE_SCOPE("native.SBT.resizeHitGroupRecords");
        hitGroupRecords_.resize(rhitSBTSize, 0);
    }

    const bool isCachedPrefixDirty = !hasValidCachedPrefix_ || cachedPrefixHitGroupIndices_ != prefixHitGroupIndices;
    if (isCachedPrefixDirty) {
        MCVR_PROFILE_SCOPE("native.SBT.rewritePrefixRecords");
        for (uint32_t i = 0; i < prefixHitGroupCount; ++i) {
            writeHitGroupRecord(hitGroupRecords_.data() + static_cast<size_t>(i) * alignedHandleSize_,
                                prefixHitGroupIndices[i]);
        }
        cachedPrefixHitGroupIndices_ = prefixHitGroupIndices;
        hasValidCachedPrefix_ = true;
    }

    const bool isCachedSuffixLayoutDirty =
        !hasValidCachedSuffix_ || cachedSuffixOffset_ != prefixHitGroupCount || cachedSuffixCount_ != cachedSuffixCount;
    const bool isCachedSuffixVersionDirty = cachedSuffixVersion_ != cachedSuffixVersion;
    const bool shouldRewriteCachedSuffix =
        isCachedSuffixLayoutDirty ||
        (isCachedSuffixVersionDirty && cachedSuffixDirtyRanges.empty() && cachedSuffixCount > 0);
    uint64_t dirtySuffixRecordCount = 0;
    for (const auto &range : cachedSuffixDirtyRanges) {
        if (range.count == 0 || range.offset >= cachedSuffixCount) { continue; }
        dirtySuffixRecordCount += std::min(range.count, cachedSuffixCount - range.offset);
    }
    MCVR_PROFILE_METRIC("sbt.cached_suffix_dirty_record_count", dirtySuffixRecordCount);
    MCVR_PROFILE_METRIC("sbt.rewrite_cached_suffix_full", shouldRewriteCachedSuffix ? 1 : 0);
    if (shouldRewriteCachedSuffix) {
        MCVR_PROFILE_SCOPE("native.SBT.rewriteSuffixRecordsFull");
        for (uint32_t i = 0; i < cachedSuffixCount; ++i) {
            writeHitGroupRecord(hitGroupRecords_.data() +
                                    static_cast<size_t>(prefixHitGroupCount + i) * alignedHandleSize_,
                                cachedSuffixHitGroupIndices[i]);
        }
    } else if (isCachedSuffixVersionDirty) {
        MCVR_PROFILE_SCOPE("native.SBT.rewriteSuffixRecordsDirty");
        for (const auto &range : cachedSuffixDirtyRanges) {
            if (range.count == 0 || range.offset >= cachedSuffixCount) { continue; }
            const uint32_t count = std::min(range.count, cachedSuffixCount - range.offset);
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t cachedIndex = range.offset + i;
                writeHitGroupRecord(hitGroupRecords_.data() +
                                        static_cast<size_t>(prefixHitGroupCount + cachedIndex) * alignedHandleSize_,
                                    cachedSuffixHitGroupIndices[cachedIndex]);
            }
        }
    }

    if (shouldRewriteCachedSuffix || isCachedSuffixVersionDirty) {
        cachedSuffixOffset_ = prefixHitGroupCount;
        cachedSuffixCount_ = cachedSuffixCount;
        cachedSuffixVersion_ = cachedSuffixVersion;
        hasValidCachedSuffix_ = true;
    }

    const bool shouldResizeBuffer = rhitSBT_ == nullptr || rhitSBT_->size() < rhitSBTSize;
    if (shouldResizeBuffer) {
        MCVR_PROFILE_SCOPE("native.SBT.resizeHitSBTBuffer");
        const VkDeviceSize capacity = std::max(rhitSBTSize, rhitSBT_ == nullptr ? VkDeviceSize(4096) :
            rhitSBT_->size() + rhitSBT_->size() / 2);
        rhitSBT_ = DeviceLocalBuffer::create(vma_, device_, true, capacity,
                                             VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                             0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, baseAlignment_);
    }

    bool uploaded = false;
    uint64_t uploadedBytes = 0;
    uint64_t uploadedRangeCount = 0;
    if (shouldResizeBuffer) {
        MCVR_PROFILE_SCOPE("native.SBT.uploadHitSBTFull");
        rhitSBT_->uploadToStagingBuffer(hitGroupRecords_.data(), rhitSBTSize, 0);
        rhitSBT_->uploadToBuffer(commandBuffer, rhitSBTSize, 0, 0);
        uploaded = true;
        uploadedBytes += rhitSBTSize;
        uploadedRangeCount++;
    } else {
        const size_t prefixByteSize = static_cast<size_t>(prefixHitGroupCount) * alignedHandleSize_;
        if (isCachedPrefixDirty && prefixByteSize > 0) {
            MCVR_PROFILE_SCOPE("native.SBT.uploadPrefixRecords");
            rhitSBT_->uploadToStagingBuffer(hitGroupRecords_.data(), prefixByteSize, 0);
            rhitSBT_->uploadToBuffer(commandBuffer, prefixByteSize, 0, 0);
            uploaded = true;
            uploadedBytes += prefixByteSize;
            uploadedRangeCount++;
        }

        if (shouldRewriteCachedSuffix && cachedSuffixCount > 0) {
            MCVR_PROFILE_SCOPE("native.SBT.uploadSuffixRecordsFull");
            const size_t suffixByteOffset = static_cast<size_t>(prefixHitGroupCount) * alignedHandleSize_;
            const size_t suffixByteSize = static_cast<size_t>(cachedSuffixCount) * alignedHandleSize_;
            rhitSBT_->uploadToStagingBuffer(hitGroupRecords_.data() + suffixByteOffset, suffixByteSize,
                                            suffixByteOffset);
            rhitSBT_->uploadToBuffer(commandBuffer, suffixByteSize, suffixByteOffset, suffixByteOffset);
            uploaded = true;
            uploadedBytes += suffixByteSize;
            uploadedRangeCount++;
        } else if (isCachedSuffixVersionDirty) {
            MCVR_PROFILE_SCOPE("native.SBT.uploadSuffixRecordsDirty");
            uint32_t uploadStart = cachedSuffixCount;
            uint32_t uploadEnd = 0;
            for (const auto &range : cachedSuffixDirtyRanges) {
                if (range.count == 0 || range.offset >= cachedSuffixCount) { continue; }
                const uint32_t count = std::min(range.count, cachedSuffixCount - range.offset);
                uploadStart = std::min(uploadStart, range.offset);
                uploadEnd = std::max(uploadEnd, range.offset + count);
            }
            if (uploadStart < uploadEnd) {
                const size_t suffixByteOffset =
                    static_cast<size_t>(prefixHitGroupCount + uploadStart) * alignedHandleSize_;
                const size_t suffixByteSize = static_cast<size_t>(uploadEnd - uploadStart) * alignedHandleSize_;
                rhitSBT_->uploadToStagingBuffer(hitGroupRecords_.data() + suffixByteOffset, suffixByteSize,
                                                suffixByteOffset);
                rhitSBT_->uploadToBuffer(commandBuffer, suffixByteSize, suffixByteOffset, suffixByteOffset);
                uploaded = true;
                uploadedBytes += suffixByteSize;
                uploadedRangeCount++;
                MCVR_PROFILE_METRIC("sbt.uploaded_dirty_span_record_count", uploadEnd - uploadStart);
            }
        }
    }
    MCVR_PROFILE_METRIC("sbt.uploaded_bytes", uploadedBytes);
    MCVR_PROFILE_METRIC("sbt.uploaded_range_count", uploadedRangeCount);

    if (uploaded) {
        MCVR_PROFILE_SCOPE("native.SBT.hitSBTUploadBarrier");
        commandBuffer->barriersBufferImage({vk::CommandBuffer::BufferMemoryBarrier{
                                               .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                               .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
                                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                               .buffer = rhitSBT_,
                                           }},
                                           {});
    }

    hitRegion_.deviceAddress = rhitSBT_->bufferAddress();
    hitRegion_.stride = alignedHandleSize_;
    hitRegion_.size = rhitSBTSize;
}

VkStridedDeviceAddressRegionKHR &vk::SBT::raygenRegion() {
    return raygenRegion_;
}

VkStridedDeviceAddressRegionKHR &vk::SBT::missRegion() {
    return missRegion_;
}

VkStridedDeviceAddressRegionKHR &vk::SBT::hitRegion() {
    return hitRegion_;
}

VkStridedDeviceAddressRegionKHR &vk::SBT::callableRegion() {
    return callableRegion_;
}
