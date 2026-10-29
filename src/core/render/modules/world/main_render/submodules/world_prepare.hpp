#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/render/chunks.hpp"
#include "core/render/modules/world/shader_pack/shader_pack.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <string>
#include <unordered_map>
#include <vector>

class Framework;
struct FrameworkContext;
class MainRenderModule;
struct ChunkGeometryMetadataCache;
struct ChunkGeometryMetadataView;
struct ChunkRenderData;
struct MainRenderModuleContext;
struct Entity;
struct EntityBatch;

struct WorldPrepareContext;

class WorldPrepare : public SharedObject<WorldPrepare> {
    friend MainRenderModule;
    friend MainRenderModuleContext;
    friend WorldPrepareContext;

  public:
    WorldPrepare();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<MainRenderModule> mainRenderModule);

    void build();

  private:
    std::weak_ptr<Framework> framework_;
    std::weak_ptr<MainRenderModule> mainRenderModule_;
    std::shared_ptr<vk::HostVisibleStagingBufferPool> stagingBufferPool_;

    std::vector<std::shared_ptr<WorldPrepareContext>> contexts_;
};

struct WorldPrepareContext : public SharedObject<WorldPrepareContext> {
    std::weak_ptr<FrameworkContext> frameworkContext;
    std::weak_ptr<MainRenderModuleContext> mainRenderModuleContext;
    std::weak_ptr<WorldPrepare> worldPrepare;

    std::shared_ptr<vk::TLAS> tlas;
    std::shared_ptr<vk::TLAS> reusableTlas;
    std::shared_ptr<vk::TLASBuilder> tlasBuilder;
    std::shared_ptr<vk::DeviceLocalBuffer> tlasStorageBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> tlasScratchBuffer;

    std::shared_ptr<vk::DescriptorTable> chunkRenderDescriptorTable;
    std::shared_ptr<vk::ComputePipeline> chunkRenderPipeline;
    std::shared_ptr<vk::HostVisibleStagingBufferPool> stagingBufferPool;
    std::shared_ptr<vk::DeviceLocalBuffer> tlasInstanceBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> blasOffsetsBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> entityGeometryInstances;
    std::shared_ptr<vk::DeviceLocalBuffer> lastIndexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastPbrBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastObjToWorldMat;
    std::shared_ptr<EntityBatch> rasterEntityBatch;
    uint32_t vistaInstanceCount = 0;
    std::vector<std::shared_ptr<ChunkRenderData>> rasterRenderableChunks;
    std::vector<ChunkRenderRecord> rasterChunkRenderRecords;
    std::vector<uint32_t> rasterChunkDrawDirtySlots;
    uint32_t rasterChunkGeometrySlotCount = 0;
    uint64_t rasterChunkHitGroupVersion = 0;
    uint64_t rasterChunkRenderVersion = 0;
    uint64_t rasterChunkDrawDirtyStartVersion = 0;
    struct ChunkHitGroupIndexInfo {
        const void *hitGroupNameMap = nullptr;
        uint32_t fallbackHitGroupIndex = 0;
        uint32_t shadowHitGroupIndex = 0;
        uint64_t chunkHitGroupVersion = 0;
        uint64_t sbtSuffixVersion = 0;
        std::vector<uint32_t> hitGroupIndices;
        std::vector<vk::SBT::HitGroupRecordRangeInfo> dirtyHitGroupRecordRanges;
    };

    std::vector<const std::string *> entityHitGroupNames;
    std::vector<const std::string *> uniqueEntityHitGroupNames;
    std::vector<uint32_t> entityHitGroupNameIds;
    std::vector<uint32_t> scratchUniqueEntityHitGroupIndices;
    std::vector<const std::string *> chunkHitGroupNames;
    std::vector<uint32_t> scratchEntityHitGroupIndices;
    std::vector<vk::SBT::HitGroupRecordRangeInfo> chunkHitGroupDirtyRanges;
    std::unordered_map<vk::SBT *, ChunkHitGroupIndexInfo> chunkHitGroupIndexInfoBySbt;
    uint64_t chunkHitGroupVersion = 0;
    uint64_t chunkHitGroupIndexVersionCounter = 0;
    bool hasValidChunkHitGroupNames = false;
    std::vector<uint32_t> scratchBlasOffsets;
    std::vector<uint64_t> scratchIndexBufferAddrs;
    std::vector<uint64_t> scratchPbrBufferAddrs;
    std::vector<uint64_t> scratchLastIndexBufferAddrs;
    std::vector<uint64_t> scratchLastPbrBufferAddrs;
    std::vector<glm::mat4> scratchLastObjToWorldMats;
    std::vector<uint8_t> scratchGeometryInstanceBytes;
    std::vector<std::pair<uint32_t, size_t>> scratchGeometryInstanceOffsets;
    uint32_t pendingTlasInstanceCount = 0;
    uint32_t tlasRefitCount = 0;
    uint64_t builtTlasChunkRenderVersion = 0;
    bool highPriorityBLASBuildSubmitted = false;
    bool accelerationStructureBuildPending = false;
    bool accelerationStructuresReady = false;

    WorldPrepareContext(std::shared_ptr<FrameworkContext> frameworkContext, std::shared_ptr<WorldPrepare> worldprepare);

    void prepareRayTracingBuffers(std::vector<VkAccelerationStructureInstanceKHR> &entityInstances,
                                  std::vector<uint32_t> &blasOffsets,
                                  std::vector<uint64_t> &indexBufferAddrs,
                                  std::vector<uint64_t> &pbrBufferAddrs,
                                  std::vector<uint64_t> &lastIndexBufferAddrs,
                                  std::vector<uint64_t> &lastPbrBufferAddrs,
                                  std::vector<glm::mat4> &lastObjToWorldMats,
                                  const ChunkGeometryMetadataView &chunkMetadata,
                                  uint32_t entityHitGroupCount,
                                  glm::dvec3 cameraPos);
    void setupHitGroupSbt(const std::unordered_map<std::string, uint32_t> &hitGroupNameToIndex,
                          uint32_t fallbackHitGroupIndex,
                          uint32_t shadowHitGroupIndex,
                          std::shared_ptr<vk::CommandBuffer> commandBuffer,
                          std::shared_ptr<vk::SBT> updateSbt,
                          std::shared_ptr<vk::SBT> querySbt);
    void submitHighPriorityBLASBuilds();
    bool prepareAccelerationStructures();
    void flushPendingHighPriorityBLASBuilds();
    void render(ShaderPackLoader::AccelerationStructureBuildMode accelerationStructureBuildMode);
};
