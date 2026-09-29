#pragma once

#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#if defined(MCVR_ENABLE_PROFILER)
#    include <array>
#endif

class Framework;
class Emission;

class Textures : public SharedObject<Textures> {
    friend class Emission;

  public:
    struct UploadRegionInfo {
        uint8_t *sourcePointer;
        uint32_t sourceSizeBytes;
        uint32_t sourceRowPixels;
        uint32_t textureId;
        int32_t sourceOffsetX;
        int32_t sourceOffsetY;
        int32_t destinationOffsetX;
        int32_t destinationOffsetY;
        uint32_t width;
        uint32_t height;
        uint32_t level;
        uint32_t uploadKind;
        uint32_t sourceBytesPerPixel;
    };

    struct UploadSubmissionInfo {
        std::shared_ptr<vk::Fence> fence;
        std::shared_ptr<vk::CommandBuffer> commandBuffer;
        std::vector<std::shared_ptr<vk::HostVisibleBuffer>> buffers;
        std::vector<std::shared_ptr<vk::DeviceLocalImage>> images;
    };

    Textures(std::shared_ptr<Framework> framework);
    ~Textures();

    void reset();
    void shutdown();
    void resetFrame();
    uint32_t allocateTexture();
    void initializeTexture(uint32_t id, uint32_t maxLevel, uint32_t width, uint32_t height, VkFormat format);
    void releaseTexture(uint32_t id);
    void queueTextureClear(uint32_t id, const VkClearColorValue &clearColor);
    void setSamplingMode(uint32_t id, VkFilter samplingMode, VkSamplerMipmapMode mipmapMode);
    void setAddressMode(uint32_t id, VkSamplerAddressMode addressMode);
    void queueUploads(const UploadRegionInfo *uploadRegions, uint32_t count);
    void performQueuedUpload(bool deferSubmission = false);
    void submitDeferredUploads();
    void bindAllTextures();
    std::shared_ptr<vk::DeviceLocalImage> texture(uint32_t id);
    std::shared_ptr<vk::Sampler> sampler(uint32_t id);
    std::shared_ptr<Emission> emission();
    void releaseEmission();

    std::map<uint32_t, std::shared_ptr<vk::DeviceLocalImage>> textures_;
    std::map<uint32_t, std::shared_ptr<vk::Sampler>> samplers;
    std::map<uint32_t, uint32_t> releasedTextureFallbackIds_;
    std::shared_ptr<Emission> emission_;
    std::mutex emissionMtx_;
    uint32_t nextID = 0;
    std::recursive_mutex mtx_;

    struct UploadPage {
        std::shared_ptr<vk::HostVisibleBuffer> buffer;
        size_t writeOffset = 0;
    };

#if defined(MCVR_ENABLE_PROFILER)
    struct UploadFrameMetrics {
        uint64_t batchCount = 0;
        uint64_t singleRegionBatchCount = 0;
        uint64_t multiRegionBatchCount = 0;
        uint64_t maxRegionsPerBatch = 0;
        std::array<uint64_t, 4> regionCounts{};
        std::array<uint64_t, 4> bytes{};
        uint64_t flushCount = 0;
        uint64_t thresholdFlushCount = 0;
        uint64_t endFrameFlushCount = 0;
        uint64_t releaseFlushCount = 0;
        uint64_t clearCount = 0;
        uint64_t stagingPageAcquireCount = 0;
        uint64_t stagingPageReuseCount = 0;
        uint64_t stagingPageDiscardCount = 0;
        uint64_t stagingPageDiscardBytes = 0;
        uint64_t stagingPageRequestedBytes = 0;
        uint64_t stagingPageCapacityBytes = 0;
        uint64_t stagingHighWaterBytes = 0;
    };
#endif

    std::map<std::pair<uint32_t, size_t>, std::vector<VkBufferImageCopy>> queuedUploadBatches_;
    std::vector<UploadPage> queuedUploadPages_;
    std::map<uint32_t, VkClearColorValue> queuedTextureClears_;
    std::vector<UploadSubmissionInfo> deferredUploadSubmissions_;
    std::vector<UploadSubmissionInfo> activeUploadSubmissions_;
    std::vector<std::shared_ptr<vk::CommandBuffer>> freeUploadCommandBuffers_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> freeUploadBuffers_;
    std::vector<std::shared_ptr<vk::Fence>> freeUploadFences_;
    size_t freeUploadBufferBytes_ = 0;
    size_t queuedUploadBytes_ = 0;
    std::shared_ptr<Framework> framework_;
    std::atomic_bool shuttingDown_{false};
#if defined(MCVR_ENABLE_PROFILER)
    size_t queuedUploadRegionCount_ = 0;
    UploadFrameMetrics uploadFrameMetrics_;
#endif

  private:
    std::shared_ptr<vk::Sampler> getOrCreateSampler(std::shared_ptr<vk::Device> device,
                                                    VkFilter samplingMode,
                                                    VkSamplerMipmapMode mipmapMode,
                                                    VkSamplerAddressMode addressMode);
    std::map<std::tuple<VkFilter, VkSamplerMipmapMode, VkSamplerAddressMode>, std::shared_ptr<vk::Sampler>>
        samplerCache_;

    static constexpr size_t UPLOAD_FLUSH_THRESHOLD = 64 * 1024 * 1024;
    static constexpr size_t UPLOAD_PAGE_SIZE = 4 * 1024 * 1024;
    static constexpr size_t UPLOAD_PAGE_GRANULARITY = 64 * 1024;
    static constexpr size_t UPLOAD_OFFSET_ALIGNMENT = 4;
    static constexpr size_t UPLOAD_STAGING_CACHE_BUDGET = UPLOAD_FLUSH_THRESHOLD;

    enum class UploadFlushReason : uint32_t {
        EndFrame = 0,
        Threshold = 1,
        TextureRelease = 2,
        TextureReinitialize = 3,
    };

    std::shared_ptr<vk::HostVisibleBuffer> acquireUploadBuffer(size_t minSizeBytes);
    void releaseUploadBuffer(std::shared_ptr<vk::HostVisibleBuffer> buffer);
    std::shared_ptr<vk::Fence> acquireUploadFence();
    std::shared_ptr<Framework> frameworkOrThrow() const;
    void ensureRunning() const;
    std::pair<size_t, size_t> appendUploadData(const UploadRegionInfo &uploadRegion, size_t bytesPerPixel);
    void recycleCompletedUploads();
    void submitDeferredUploadsLocked();
    void flushQueuedUploads(UploadFlushReason reason, bool deferSubmission = false);
#if defined(MCVR_ENABLE_PROFILER)
    void emitAndResetUploadFrameMetrics();
#endif
};
