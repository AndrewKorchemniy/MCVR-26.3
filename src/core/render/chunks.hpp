#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/build_geometry.hpp"
#include "core/render/emission.hpp"
#include "core/render/world.hpp"
#include "core/render/vista.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class Framework;
class FrameResourceRetainer;

struct ChunkGpuBufferPoolStats {
    size_t entryCount = 0;
    VkDeviceSize capacityBytes = 0;
};

class ChunkGpuBufferPool : public SharedObject<ChunkGpuBufferPool> {
  public:
    std::shared_ptr<vk::DeviceLocalBuffer> acquire(const std::shared_ptr<vk::VMA> &vma,
                                                   const std::shared_ptr<vk::Device> &device,
                                                   uint64_t frameSerial,
                                                   VkDeviceSize requiredBytes,
                                                   VkBufferUsageFlags usage,
                                                   bool persistStaging,
                                                   VmaMemoryUsage memoryUsage,
                                                   VkDeviceSize alignment = 16);
    void release(std::shared_ptr<vk::DeviceLocalBuffer> buffer, uint64_t reusableFrameSerial);
    void sweep(uint64_t frameSerial, bool aggressive = false);
    void clear();
    bool hasEntries() const;
    bool markSweepQueued();
    void clearSweepQueued();
    ChunkGpuBufferPoolStats stats() const;

  private:
    struct Entry {
        std::shared_ptr<vk::DeviceLocalBuffer> buffer;
        uint64_t reusableFrameSerial = 0;
    };

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
    std::atomic<size_t> entryCount_ = 0;
    std::atomic<bool> sweepQueued_ = false;
};

struct ChunkPackedData {
    int32_t x;
    int32_t y;
    int32_t z;
    uint32_t geometryCount;
    uint32_t lightCount = 0;
    VkDeviceAddress lightBufferAddress = 0;
    uint32_t entityLightCount = 0;
    VkDeviceAddress entityLightBufferAddress = 0;
    uint32_t globalEntityLightCount = 0;
    VkDeviceAddress globalEntityLightBufferAddress = 0;
};
static_assert(sizeof(ChunkPackedData) == 64);

struct ChunkEntityLightRange {
    uint32_t chunkIndex = 0;
    uint32_t lightCount = 0;
    VkDeviceAddress lightBufferAddress = 0;
    glm::ivec3 chunkOrigin = glm::ivec3(0);
};

struct ChunkRenderRecord {
    glm::ivec4 positionAndGeometryCount;
    glm::uvec4 offsetsMaskFlags;
    glm::uvec4 blasAddress;
};
static_assert(sizeof(ChunkRenderRecord) == 48);

struct ChunkBuildTask {
    int x = 0, y = 0, z = 0;
    int64_t generation = 0;
    std::shared_ptr<const VistaTerrain> terrain;
    int64_t id;
    int geometryCount;
    int *geometryTypes;
    const char **geometryGroupNames;
    int *geometryTextures;
    int *vertexFormats;
    int *indexTypes;
    int *vertexCounts;
    vk::VertexFormat::ChunkPBRVertex **vertices;
    void **indices;
    std::shared_ptr<BuildBufferLease> transferredBufferLease;
    bool isImportant;
};

struct ChunkBuildData : public SharedObject<ChunkBuildData> {
    int64_t id;
    int x, y, z;
    int64_t version;
    int64_t generation = 0;
    std::shared_ptr<const VistaTerrain> terrain;
    bool collectChunkEmission = false;
    bool important = false;
    uint32_t allVertexCount;
    uint32_t allIndexCount;
    uint32_t geometryCount;
    std::vector<World::GeometryTypes> geometryTypes;
    std::vector<std::string> geometryGroupNames;
    std::vector<ChunkBuildGeometry> geometries;
    std::vector<VkDeviceAddress> indexBufferAddresses;
    std::vector<VkDeviceAddress> pbrBufferAddresses;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> blasStorageBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> blasScratchBuffer;
    std::shared_ptr<vk::BLAS> blas;
    std::shared_ptr<vk::BLASBuilder> blasBuilder;
    std::shared_ptr<ChunkGpuBufferPool> indexBufferPool;
    std::shared_ptr<ChunkGpuBufferPool> pbrBufferPool;
    std::shared_ptr<ChunkGpuBufferPool> blasStoragePool;
    std::shared_ptr<ChunkGpuBufferPool> blasScratchPool;
    std::vector<LightInfo> lightInfos;
    std::shared_ptr<vk::DeviceLocalBuffer> lightBuffer;
    uint32_t lightCount = 0;

    ChunkBuildData(int64_t id,
                   int x,
                   int y,
                   int z,
                   int64_t version,
                   bool collectChunkEmission,
                   bool important,
                   uint32_t allVertexCount,
                   uint32_t allIndexCount,
                   uint32_t geometryCount,
                   std::vector<World::GeometryTypes> &&geometryTypes,
                   std::vector<std::string> &&geometryGroupNames,
                   std::vector<ChunkBuildGeometry> &&geometries);

    void buildLightInfos(const Emission &emission);
    void buildLightBuffer(const std::shared_ptr<vk::VMA> &vma,
                          const std::shared_ptr<vk::Device> &device,
                          bool persistStaging = true);
    void build(bool persistStaging = true);
    void releaseBuildInputs();
};

struct Chunk1;
struct ChunkRenderData;

struct ChunkMetadataUpdateRangeInfo {
    uint32_t offset = 0;
    uint32_t count = 0;
};

struct ChunkMetadataSlabInfo {
    uint32_t geometryOffset = 0;
    uint32_t hitGroupOffset = 0;
};

struct ChunkMetadataSlabAllocationInfo {
    uint32_t geometryOffset = 0;
    uint32_t hitGroupOffset = 0;
    uint32_t geometryCapacity = 0;
    uint32_t geometryCount = 0;

    bool allocated() const {
        return geometryCapacity > 0;
    }
};

struct ChunkGeometryMetadataCache {
    std::vector<const std::string *> hitGroupNames;
    std::vector<VkDeviceAddress> indexBufferAddresses;
    std::vector<VkDeviceAddress> pbrBufferAddresses;
    std::vector<ChunkRenderRecord> renderRecords;
    std::vector<ChunkMetadataSlabAllocationInfo> chunkAllocations;
    std::map<uint32_t, std::vector<ChunkMetadataSlabInfo>> freeSlabsByCapacity;
    std::vector<ChunkMetadataUpdateRangeInfo> renderRecordDirtyRanges;
    std::vector<ChunkMetadataUpdateRangeInfo> geometryDirtyRanges;
    std::vector<ChunkMetadataUpdateRangeInfo> hitGroupDirtyRanges;
    std::vector<uint32_t> rasterDrawDirtySlots;
    std::vector<uint8_t> rasterDrawDirtySlotMarks;
    std::shared_ptr<vk::DeviceLocalBuffer> renderRecordBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBufferAddressBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBufferAddressBuffer;
    size_t geometrySlotCount = 0;
    size_t actualGeometryCount = 0;
    uint64_t hitGroupVersion = 0;
    uint64_t renderVersion = 0;
    uint64_t rasterDrawDirtyStartVersion = 0;
    bool hasPendingGpuUpload = false;

    void reset(uint32_t chunkCount);
    void clear();
    void updateChunk(uint32_t chunkId, const Chunk1 &chunk, uint32_t renderSlot);
    void removeChunk(uint32_t chunkId);
    void updateRenderRecord(uint32_t chunkId, const Chunk1 &chunk, uint32_t renderSlot);
    void resizeRenderRecords(size_t renderRecordCount);
    void markRasterDrawDirtySlot(uint32_t renderSlot);
    void clearRasterDrawDirtySlots();
};

// The part of ChunkGeometryMetadataCache that frame recording needs once the chunk lock is released: counts and the
// GPU buffers. Taking this instead of a full copy avoids copying every metadata vector each frame.
struct ChunkGeometryMetadataView {
    size_t renderRecordCount = 0;
    size_t geometrySlotCount = 0;
    size_t actualGeometryCount = 0;
    std::shared_ptr<vk::DeviceLocalBuffer> renderRecordBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBufferAddressBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBufferAddressBuffer;
};

struct ChunkBuildDataBatch: public SharedObject<ChunkBuildDataBatch> {
    std::vector<std::shared_ptr<ChunkBuildData>> batchData;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> blasStorageBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> blasScratchBuffer;
    std::shared_ptr<vk::BLASBatchBuilder> blasBatchBuilder;
    std::vector<std::shared_ptr<vk::BLASBuilder>> blasBuilders;

    ChunkBuildDataBatch(std::vector<std::shared_ptr<ChunkBuildData>> &&batchData);
    ChunkBuildDataBatch(uint32_t maxBatchSize,
                        std::set<int64_t> &queuedIndex,
                        std::vector<std::shared_ptr<Chunk1>> &chunks,
                        std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                        glm::vec3 cameraPos);
    void build();
};

class ChunkBuildScheduler : public SharedObject<ChunkBuildScheduler> {
  public:
    ChunkBuildScheduler(std::set<int64_t> &queuedIndex,
                        std::vector<std::shared_ptr<Chunk1>> &chunks,
                        std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                        std::recursive_mutex &mutex,
                        std::vector<ChunkPackedData> &chunkPackedData,
                        std::vector<std::shared_ptr<Chunk1>> &renderableChunks,
                        std::vector<std::shared_ptr<ChunkRenderData>> &renderableChunkSnapshots,
                        std::vector<int64_t> &renderableChunkIds,
                        std::vector<uint32_t> &renderableChunkSlots,
                        ChunkGeometryMetadataCache &chunkGeometryMetadataCache,
                        uint32_t chunkBuildingBatchSize,
                        uint32_t chunkBuildingTotalBatches,
                        uint32_t chunkBuildingThreads);
    ~ChunkBuildScheduler();

    void tryCheckBatchesFinish();
    void waitAllBatchesFinish();
    void tryScheduleBatches(uint32_t maxBatchSize);

    uint32_t chunkBuildingBatchSize();
    uint32_t chunkBuildingTotalBatches();

  private:
    struct CpuBuildJob {
        std::shared_ptr<vk::Fence> fence;
        std::shared_ptr<vk::CommandBuffer> commandBuffer;
        std::shared_ptr<ChunkBuildDataBatch> chunkBuildDataBatch;
        std::exception_ptr exception;
    };

    void publishBatchLocked(const std::shared_ptr<ChunkBuildDataBatch> &chunkBuildDataBatch);
    void submitBatch(std::shared_ptr<ChunkBuildDataBatch> chunkBuildDataBatch,
                     std::shared_ptr<vk::Fence> fence,
                     std::shared_ptr<vk::CommandBuffer> commandBuffer);
    void startCpuBuildWorkers();
    void stopCpuBuildWorkers();
    void cpuBuildWorkerMain();
    void enqueueCpuBuildJob(CpuBuildJob job);
    void submitReadyCpuBuildBatches(uint32_t &batchBudget, bool wait);
    bool hasCpuBuildWorkLocked() const;
    bool hasCurrentBuildDataLocked(const std::shared_ptr<ChunkBuildDataBatch> &chunkBuildDataBatch) const;
    bool isBuildDataCurrentLocked(const std::shared_ptr<ChunkBuildData> &chunkBuildData) const;
    void pruneStalePendingBatchDataLocked();

    std::set<int64_t> &queuedIndex_;
    std::vector<std::shared_ptr<Chunk1>> &chunks_;
    std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas_;
    std::recursive_mutex &mutex_;
    std::vector<ChunkPackedData> &chunkPackedData_;
    std::vector<std::shared_ptr<Chunk1>> &renderableChunks_;
    std::vector<std::shared_ptr<ChunkRenderData>> &renderableChunkSnapshots_;
    std::vector<int64_t> &renderableChunkIds_;
    std::vector<uint32_t> &renderableChunkSlots_;
    ChunkGeometryMetadataCache &chunkGeometryMetadataCache_;

    std::queue<std::shared_ptr<vk::Fence>> freeFences_;
    std::queue<std::shared_ptr<vk::CommandBuffer>> freeCommandBuffers_;
    std::list<std::shared_ptr<vk::Fence>> buildingFences_;
    std::list<std::shared_ptr<vk::CommandBuffer>> buildingCommandBuffers_;
    std::list<std::shared_ptr<ChunkBuildDataBatch>> buildingBatches_;
    std::deque<CpuBuildJob> pendingCpuBuildJobs_;
    std::deque<CpuBuildJob> readyCpuBuildJobs_;
    std::vector<std::thread> cpuBuildWorkers_;
    mutable std::mutex cpuBuildMutex_;
    std::condition_variable cpuBuildCondition_;
    uint32_t activeCpuBuildJobCount_ = 0;
    bool stopCpuBuildWorkers_ = false;
    std::vector<std::shared_ptr<ChunkBuildData>> pendingBatchData_;
    uint32_t pendingBatchFrames_ = 0;
    bool useSecondaryQueue_ = false;
    static constexpr uint32_t maxPendingBatchFrames_ = 3;

    uint32_t chunkBuildingBatchSize_;
    uint32_t chunkBuildingTotalBatches_;
    uint32_t chunkBuildingThreads_;
};

struct ChunkRenderData : public SharedObject<ChunkRenderData> {
    const Chunk1 *sourceChunk = nullptr;
    int x, y, z;
    int64_t latestVersion = 0;
    int64_t blasVersion = -1;
    std::shared_ptr<vk::BLAS> blas;
    std::shared_ptr<std::vector<VkDeviceAddress>> indexBufferAddresses;
    std::shared_ptr<std::vector<VkDeviceAddress>> pbrBufferAddresses;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> blasStorageBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> blasScratchBuffer;
    std::shared_ptr<std::vector<LightInfo>> lightInfos;
    std::shared_ptr<vk::DeviceLocalBuffer> lightBuffer;
    uint32_t lightCount = 0;
    uint32_t geometryCount;
    std::shared_ptr<std::vector<std::string>> geometryGroupNames;
    std::shared_ptr<std::vector<uint32_t>> vertexCounts;
    std::shared_ptr<std::vector<uint32_t>> indexCounts;
};

struct Chunk1 : public SharedObject<Chunk1> {
    constexpr static float T_HALF = 200;
    constexpr static float T_WEIGHT = 1.0;

    constexpr static float D_HALF = 128;
    constexpr static float D_SENSITIVITY = 1.35;
    constexpr static float D_PRIORITY_FLOOR = 0.4;
    constexpr static float D_WEIGHT = 1.1;

    int x = 0, y = 0, z = 0;
    int64_t generation = 0;
    int64_t latestVersion = 0;
    bool terrainEmpty = false;
    std::shared_ptr<const VistaTerrain> terrain;
    std::chrono::steady_clock::time_point lastUpdate;

    std::shared_ptr<vk::BLAS> blas;
    int64_t blasVersion = -1;
    std::shared_ptr<std::vector<VkDeviceAddress>> indexBufferAddresses;
    std::shared_ptr<std::vector<VkDeviceAddress>> pbrBufferAddresses;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> blasStorageBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> blasScratchBuffer;
    std::shared_ptr<ChunkGpuBufferPool> indexBufferPool;
    std::shared_ptr<ChunkGpuBufferPool> pbrBufferPool;
    std::shared_ptr<ChunkGpuBufferPool> blasStoragePool;
    std::shared_ptr<ChunkGpuBufferPool> blasScratchPool;
    std::shared_ptr<ChunkGpuBufferPool> activeIndexBufferPool;
    std::shared_ptr<ChunkGpuBufferPool> activePbrBufferPool;
    std::shared_ptr<ChunkGpuBufferPool> activeBlasStoragePool;
    std::shared_ptr<std::vector<LightInfo>> lightInfos;
    std::shared_ptr<vk::DeviceLocalBuffer> lightBuffer;
    uint32_t lightCount = 0;
    uint32_t geometryCount;
    std::shared_ptr<std::vector<std::string>> geometryGroupNames;
    std::shared_ptr<std::vector<uint32_t>> vertexCounts;
    std::shared_ptr<std::vector<uint32_t>> indexCounts;
    std::shared_ptr<std::vector<std::string>> stableHitGroupNames;
    std::vector<const std::string *> hitGroupNameRefs;

    float buildFactor(std::chrono::steady_clock::time_point currentTime, glm::vec3 cameraPos, glm::vec3 chunkPos);

    bool enqueue(std::shared_ptr<ChunkBuildData> chunkBuildData);
    void invalidate();
    void retainResources(FrameResourceRetainer &frr);
    void releaseEmissionResources(FrameResourceRetainer &frr);
    std::shared_ptr<ChunkRenderData> tryGetValid();
    const std::shared_ptr<ChunkRenderData> &renderDataSnapshot() const;

  private:
    void rebuildRenderDataSnapshot();

    std::shared_ptr<ChunkRenderData> renderDataSnapshot_;
};

class Chunks : public SharedObject<Chunks> {
    friend World;

  public:
    Chunks(std::shared_ptr<Framework> framework);

    void reset(uint32_t numChunks, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, int32_t bottomSectionCoord);
    void resetScheduler();
    void resetFrame();
    void invalidateChunk(int id, int x, int y, int z, int64_t generation,
                         std::shared_ptr<const VistaTerrain> terrain);
    void relocateChunk(int id, int x, int y, int z, int64_t generation);
    void queueChunkBuild(ChunkBuildTask task);
    void setCollectChunkEmission(bool collect);
    void trackChunkBufferPoolForSweep(const std::shared_ptr<ChunkGpuBufferPool> &pool);
    uint64_t frameSerial() const;
    uint64_t bufferReuseDelayFrames() const;

    bool isChunkReady(int64_t id);

    void close();

    std::recursive_mutex &mutex();
    std::vector<std::shared_ptr<Chunk1>> &chunks();
    const std::vector<std::shared_ptr<Chunk1>> &renderableChunks();
    const std::vector<std::shared_ptr<ChunkRenderData>> &renderableChunkSnapshots();
    const ChunkGeometryMetadataCache &chunkGeometryMetadataCache();
    void clearChunkGeometryHitGroupDirtyRanges();
    void clearChunkRasterDrawDirtySlots();
    void uploadChunkGeometryMetadataCache(std::shared_ptr<vk::CommandBuffer> commandBuffer);
    void updateEntityLightData(const std::vector<ChunkEntityLightRange> &ranges,
                               uint32_t globalLightCount,
                               VkDeviceAddress globalLightBufferAddress);
    void queueChunkPackedDataUpload();
    std::shared_ptr<ChunkBuildScheduler> chunkBuildScheduler();
    std::shared_ptr<std::vector<std::shared_ptr<vk::BLASBuilder>>> importantBLASBuilders();
    std::shared_ptr<vk::DeviceLocalBuffer> chunkPackedData();
    glm::ivec4 chunkGridInfo();
    void setChunkStorageSectionPos(glm::ivec3 sectionPos);
    glm::ivec4 chunkStorageSectionPos();

  private:
    struct ChunkFrameBufferPools {
        std::shared_ptr<ChunkGpuBufferPool> indexBufferPool;
        std::shared_ptr<ChunkGpuBufferPool> pbrBufferPool;
        std::shared_ptr<ChunkGpuBufferPool> blasStoragePool;
        std::shared_ptr<ChunkGpuBufferPool> blasScratchPool;
    };

    void allocateChunkPackedDataBuffers();
    void releaseEmissionResources();
    void clearEntityLightDataLocked();
    void uploadChunkPackedDataForCurrentFrameLocked();
    void ensureImportantFrameBufferPools();
    ChunkFrameBufferPools &importantFrameBufferPoolsForCurrentFrame();
    bool isBuildDataCurrentLocked(const std::shared_ptr<ChunkBuildData> &chunkBuildData) const;
    void queueImportantBuildUploadsLocked(const std::shared_ptr<ChunkBuildData> &chunkBuildData);
    void publishImportantBuildDataLocked(const std::shared_ptr<ChunkBuildData> &chunkBuildData);
    void rebuildRenderableChunkSnapshotsLocked();
    void recordChunkMemoryBreakdownLocked() const;

    std::recursive_mutex mutex_;
    std::vector<std::shared_ptr<Chunk1>> chunks_;
    std::vector<std::shared_ptr<Chunk1>> renderableChunks_;
    std::vector<std::shared_ptr<ChunkRenderData>> renderableChunkSnapshots_;
    std::vector<int64_t> renderableChunkIds_;
    std::vector<uint32_t> renderableChunkSlots_;
    ChunkGeometryMetadataCache chunkGeometryMetadataCache_;
    std::vector<ChunkPackedData> chunkPackedData_;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> chunkPackedDataBuffers_;
    std::vector<uint32_t> entityLightDataChunkIndices_;
    bool chunkPackedDataDirty_ = false;
    std::vector<std::shared_ptr<ChunkBuildData>> chunkBuildDatas_;
    std::set<int64_t> queuedIndex_;
    std::shared_ptr<ChunkBuildScheduler> chunkBuildScheduler_;
    std::vector<ChunkFrameBufferPools> importantFrameBufferPools_;
    std::vector<std::shared_ptr<ChunkGpuBufferPool>> activeChunkBufferPools_;

    std::shared_ptr<std::vector<std::shared_ptr<vk::BLASBuilder>>> importantBLASBuilders_;
    uint64_t frameSerial_ = 0;
    bool memoryPressureActive_ = false;
    int32_t sizeX_ = 0;
    int32_t sizeY_ = 0;
    int32_t sizeZ_ = 0;
    int32_t bottomSectionCoord_ = 0;
    glm::ivec3 chunkStorageSectionPos_ = glm::ivec3(0);
};
