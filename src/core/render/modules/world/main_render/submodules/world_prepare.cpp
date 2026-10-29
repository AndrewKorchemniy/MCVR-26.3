#include "core/render/vista.hpp"
#include "core/render/modules/world/main_render/submodules/world_prepare.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/modules/world/main_render/main_render_module.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string_view>
#include <glm/gtc/type_ptr.hpp>

static const std::string shadowHitGroupName = "shadow";
static const std::string defaultHitGroupName = "default";
static constexpr size_t worldPrepareStagingCacheBudgetBytes = 32ull * 1024ull * 1024ull;

struct ChunkRenderPushConstants {
    glm::ivec4 cameraBase;
    glm::vec4 cameraFraction;
    glm::ivec4 lastCameraBase;
    glm::vec4 lastCameraFraction;
    uint32_t chunkCount;
    uint32_t entityInstanceCount;
    uint32_t entityGeometryCount;
    uint32_t entityHitGroupCount;
};
static_assert(sizeof(ChunkRenderPushConstants) == 80);

static size_t bufferByteSize(size_t byteSize) {
    return std::max<size_t>(byteSize, 16);
}

static std::shared_ptr<vk::DeviceLocalBuffer>
createFrameBuffer(const std::shared_ptr<vk::VMA> &vma,
                  const std::shared_ptr<vk::Device> &device,
                  const std::shared_ptr<vk::HostVisibleStagingBufferPool> &stagingPool,
                  size_t byteSize,
                  VkBufferUsageFlags usage,
                  VkDeviceSize alignment = 16) {
    auto buffer = vk::DeviceLocalBuffer::create(vma, device, false, bufferByteSize(byteSize), usage, 0,
                                                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, alignment);
    buffer->setStagingBufferPool(stagingPool);
    return buffer;
}

static void ensureFrameBufferCapacity(std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                                      const std::shared_ptr<vk::VMA> &vma,
                                      const std::shared_ptr<vk::Device> &device,
                                      const std::shared_ptr<vk::HostVisibleStagingBufferPool> &stagingPool,
                                      FrameResourceRetainer &frr,
                                      size_t byteSize,
                                      VkBufferUsageFlags usage,
                                      VkDeviceSize alignment = 16) {
    const size_t requiredByteSize = bufferByteSize(byteSize);
    if (buffer != nullptr && buffer->size() >= requiredByteSize) { return; }

    frr.retain(buffer);
    buffer = createFrameBuffer(vma, device, stagingPool, requiredByteSize, usage, alignment);
}

template <typename T>
static void uploadVectorRange(const std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                              const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
                              const std::vector<T> &data,
                              size_t elementOffset = 0) {
    if (data.empty()) { return; }

    const size_t byteSize = data.size() * sizeof(T);
    const size_t byteOffset = elementOffset * sizeof(T);
    buffer->uploadToStagingBuffer(const_cast<T *>(data.data()), byteSize, byteOffset);
    buffer->uploadToBuffer(commandBuffer, byteSize, byteOffset, byteOffset);
}

static void copyBufferRange(const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
                            const std::shared_ptr<vk::DeviceLocalBuffer> &src,
                            const std::shared_ptr<vk::DeviceLocalBuffer> &dst,
                            size_t byteSize,
                            size_t srcOffset,
                            size_t dstOffset) {
    if (byteSize == 0 || src == nullptr || dst == nullptr) { return; }

    VkBufferCopy copyRegion{
        .srcOffset = srcOffset,
        .dstOffset = dstOffset,
        .size = byteSize,
    };
    vkCmdCopyBuffer(commandBuffer->vkCommandBuffer(), src->vkBuffer(), dst->vkBuffer(), 1, &copyRegion);
}

static void resolveHitGroupIndices(const std::vector<const std::string *> &hitGroupNames,
                                   const std::unordered_map<std::string, uint32_t> &hitGroupNameToIndex,
                                   uint32_t fallbackHitGroupIndex,
                                   uint32_t shadowHitGroupIndex,
                                   std::vector<uint32_t> &hitGroupIndices) {
    hitGroupIndices.clear();
    hitGroupIndices.reserve(hitGroupNames.size());

    for (const std::string *groupName : hitGroupNames) {
        if (groupName == nullptr) {
            hitGroupIndices.push_back(fallbackHitGroupIndex);
            continue;
        }
        if (*groupName == shadowHitGroupName) {
            hitGroupIndices.push_back(shadowHitGroupIndex);
            continue;
        }

        auto iter = hitGroupNameToIndex.find(*groupName);
        hitGroupIndices.push_back(iter == hitGroupNameToIndex.end() ? fallbackHitGroupIndex : iter->second);
    }
}

static uint32_t resolveHitGroupIndex(const std::string *groupName,
                                     const std::unordered_map<std::string, uint32_t> &hitGroupNameToIndex,
                                     uint32_t fallbackHitGroupIndex,
                                     uint32_t shadowHitGroupIndex) {
    if (groupName == nullptr) { return fallbackHitGroupIndex; }
    if (*groupName == shadowHitGroupName) { return shadowHitGroupIndex; }

    auto iter = hitGroupNameToIndex.find(*groupName);
    return iter == hitGroupNameToIndex.end() ? fallbackHitGroupIndex : iter->second;
}

static void
mergeHitGroupRecordRange(std::vector<vk::SBT::HitGroupRecordRangeInfo> &ranges, uint32_t offset, uint32_t count) {
    if (count == 0) { return; }

    const uint32_t end = offset + count;
    for (auto &range : ranges) {
        const uint32_t rangeEnd = range.offset + range.count;
        if (end < range.offset || offset > rangeEnd) { continue; }
        const uint32_t mergedOffset = std::min(range.offset, offset);
        const uint32_t mergedEnd = std::max(rangeEnd, end);
        range.offset = mergedOffset;
        range.count = mergedEnd - mergedOffset;
        return;
    }
    ranges.push_back({.offset = offset, .count = count});
}

static void copyDirtyHitGroupNameRanges(std::vector<const std::string *> &dst,
                                        const std::vector<const std::string *> &src,
                                        const std::vector<ChunkMetadataUpdateRangeInfo> &ranges,
                                        std::vector<vk::SBT::HitGroupRecordRangeInfo> &copiedRanges) {
    copiedRanges.clear();
    for (const auto &range : ranges) {
        if (range.count == 0 || range.offset >= src.size()) { continue; }

        const uint32_t count = static_cast<uint32_t>(std::min<size_t>(range.count, src.size() - range.offset));
        std::copy(src.begin() + range.offset, src.begin() + range.offset + count, dst.begin() + range.offset);
        mergeHitGroupRecordRange(copiedRanges, range.offset, count);
    }
}

static void resolveDirtyHitGroupIndexRanges(const std::vector<const std::string *> &hitGroupNames,
                                            const std::unordered_map<std::string, uint32_t> &hitGroupNameToIndex,
                                            uint32_t fallbackHitGroupIndex,
                                            uint32_t shadowHitGroupIndex,
                                            const std::vector<vk::SBT::HitGroupRecordRangeInfo> &ranges,
                                            std::vector<uint32_t> &hitGroupIndices,
                                            std::vector<vk::SBT::HitGroupRecordRangeInfo> &resolvedRanges) {
    resolvedRanges.clear();
    for (const auto &range : ranges) {
        if (range.count == 0 || range.offset >= hitGroupNames.size()) { continue; }

        const uint32_t count =
            static_cast<uint32_t>(std::min<size_t>(range.count, hitGroupNames.size() - range.offset));
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t index = range.offset + i;
            hitGroupIndices[index] = resolveHitGroupIndex(hitGroupNames[index], hitGroupNameToIndex,
                                                          fallbackHitGroupIndex, shadowHitGroupIndex);
        }
        mergeHitGroupRecordRange(resolvedRanges, range.offset, count);
    }
}

WorldPrepare::WorldPrepare() {}

void WorldPrepare::init(std::shared_ptr<Framework> framework, std::shared_ptr<MainRenderModule> mainRenderModule) {
    framework_ = framework;
    mainRenderModule_ = mainRenderModule;
    stagingBufferPool_ = vk::HostVisibleStagingBufferPool::create(framework->vma(), worldPrepareStagingCacheBudgetBytes,
                                                                  vk::ResourceMemoryCategory::Other);
}

void WorldPrepare::build() {
    auto framework = framework_.lock();
    auto mainRenderModule = mainRenderModule_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        contexts_[i] = WorldPrepareContext::create(framework->contexts()[i], shared_from_this());
    }
}

WorldPrepareContext::WorldPrepareContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                         std::shared_ptr<WorldPrepare> worldPrepare)
    : frameworkContext(frameworkContext), worldPrepare(worldPrepare) {
    VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(ChunkRenderPushConstants),
    };

    vk::DescriptorTableBuilder builder;
    builder.definePushConstant(pushConstantRange);
    auto &set0 = builder.beginDescriptorLayoutSet();
    auto &set0Bindings = set0.beginDescriptorLayoutSetBinding();
    for (uint32_t binding = 0; binding < 6; binding++) {
        set0Bindings.defineDescriptorLayoutSetBinding({
            .binding = binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        });
    }
    set0Bindings.defineDescriptorLayoutSetBinding({
        .binding = 6,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    });
    set0Bindings.endDescriptorLayoutSetBinding();
    set0.endDescriptorLayoutSet();

    auto device = frameworkContext->device;
    chunkRenderDescriptorTable = builder.build(device);
    auto shader = vk::Shader::create(
        device, (Renderer::folderPath / "shaders/world/ray_tracing/chunk_render_state_comp.spv").string());
    chunkRenderPipeline = vk::ComputePipelineBuilder{}
                              .defineShader(shader)
                              .definePipelineLayout(chunkRenderDescriptorTable)
                              .build(device);
    stagingBufferPool = worldPrepare->stagingBufferPool_;
}

void WorldPrepareContext::prepareRayTracingBuffers(std::vector<VkAccelerationStructureInstanceKHR> &entityInstances,
                                                   std::vector<uint32_t> &blasOffsets,
                                                   std::vector<uint64_t> &indexBufferAddrs,
                                                   std::vector<uint64_t> &pbrBufferAddrs,
                                                   std::vector<uint64_t> &lastIndexBufferAddrs,
                                                   std::vector<uint64_t> &lastPbrBufferAddrs,
                                                   std::vector<glm::mat4> &lastObjToWorldMats,
                                                   const ChunkGeometryMetadataView &chunkMetadata,
                                                   uint32_t entityHitGroupCount,
                                                   glm::dvec3 cameraPos) {
    MCVR_PROFILE_SCOPE("native.WorldPrepareContext.prepareRayTracingBuffers");
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto &frameResourceRetainer = framework->frameResourceRetainer();
    auto mainQueueIndex = physicalDevice->mainQueueIndex();
    auto cmdBuffer = context->worldCommandBuffer;

    const uint32_t entityInstanceCount = static_cast<uint32_t>(entityInstances.size());
    const uint32_t entityGeometryCount = static_cast<uint32_t>(indexBufferAddrs.size());
    const uint32_t chunkInstanceCount = static_cast<uint32_t>(chunkMetadata.renderRecordCount);
    const uint32_t chunkGeometrySlotCount = static_cast<uint32_t>(chunkMetadata.geometrySlotCount);
    const uint32_t totalInstanceCount = entityInstanceCount + chunkInstanceCount;
    const uint32_t totalGeometryCount = entityGeometryCount + chunkGeometrySlotCount;
    MCVR_PROFILE_METRIC("world_prepare.entity_instances", entityInstanceCount);
    MCVR_PROFILE_METRIC("world_prepare.entity_geometries", entityGeometryCount);
    MCVR_PROFILE_METRIC("world_prepare.chunk_instances", chunkInstanceCount);
    MCVR_PROFILE_METRIC("world_prepare.chunk_geometry_slots", chunkGeometrySlotCount);
    MCVR_PROFILE_METRIC("world_prepare.total_instances", totalInstanceCount);
    MCVR_PROFILE_METRIC("world_prepare.total_geometries", totalGeometryCount);
    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.ensureRayTracingBufferCapacity");
        ensureFrameBufferCapacity(tlasInstanceBuffer, vma, device, stagingBufferPool, frameResourceRetainer,
                                  totalInstanceCount * sizeof(VkAccelerationStructureInstanceKHR),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        ensureFrameBufferCapacity(blasOffsetsBuffer, vma, device, stagingBufferPool, frameResourceRetainer,
                                  totalInstanceCount * sizeof(uint32_t),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        ensureFrameBufferCapacity(lastObjToWorldMat, vma, device, stagingBufferPool, frameResourceRetainer,
                                  totalInstanceCount * sizeof(glm::mat4),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        ensureFrameBufferCapacity(indexBufferAddr, vma, device, stagingBufferPool, frameResourceRetainer,
                                  totalGeometryCount * sizeof(uint64_t),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        ensureFrameBufferCapacity(pbrBufferAddr, vma, device, stagingBufferPool, frameResourceRetainer,
                                  totalGeometryCount * sizeof(uint64_t),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        ensureFrameBufferCapacity(lastIndexBufferAddr, vma, device, stagingBufferPool, frameResourceRetainer,
                                  totalGeometryCount * sizeof(uint64_t),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        ensureFrameBufferCapacity(lastPbrBufferAddr, vma, device, stagingBufferPool, frameResourceRetainer,
                                  totalGeometryCount * sizeof(uint64_t),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    using InstanceHeader = vk::VertexFormat::EntityGeometryHeader;
    using FullInstance = vk::VertexFormat::EntityGeometryInstance;
    auto &instanceBytes = scratchGeometryInstanceBytes;
    auto &instanceOffsets = scratchGeometryInstanceOffsets;
    instanceBytes.clear();
    instanceOffsets.clear();
    instanceOffsets.reserve(entityGeometryCount);
    size_t denseMaterialCount = 0;
    if (rasterEntityBatch != nullptr) {
        size_t addressIndex = 0;
        for (const auto &entity : rasterEntityBatch->entities) {
            for (uint32_t geometry = 0; geometry < entity->geometryCount; ++geometry, ++addressIndex) {
                if (entity->prebuiltBLAS < 0) continue;
                const auto address = pbrBufferAddrs[addressIndex];
                const InstanceHeader header{static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32),
                    static_cast<uint32_t>(addressIndex), 0, entity->material};
                const size_t offset = instanceBytes.size();
                const bool full = (entity->material.flags & 6u) != 0;
                instanceBytes.resize(offset + (full ? sizeof(FullInstance) : sizeof(InstanceHeader)));
                std::memcpy(instanceBytes.data() + offset, &header, sizeof(header));
                if (full) {
                    FullInstance record{};
                    std::memcpy(&record, &header, sizeof(header));
                    glm::mat3 toObject = glm::transpose(glm::mat3(entity->localTransform));
                    glm::mat3 toWorld = glm::inverse(toObject);
                    for (uint32_t col = 0; col < 3; ++col) {
                        record.normalToObject[col] = glm::vec4(toObject[col], 0);
                        record.normalToWorld[col] = glm::vec4(toWorld[col], 0);
                    }
                    if ((entity->material.flags & 2u) != 0 && entity->attributeTransform != nullptr) {
                        ++denseMaterialCount;
                        const auto rawAddress = entity->sharedGeometry->programBuffer->bufferAddress();
                        record.rawAddressLo = static_cast<uint32_t>(rawAddress);
                        record.rawAddressHi = static_cast<uint32_t>(rawAddress >> 32);
                        std::memcpy(record.attributes, entity->attributeTransform->data(), sizeof(record.attributes));
                    }
                    std::memcpy(instanceBytes.data() + offset, &record, sizeof(record));
                }
                instanceOffsets.emplace_back(static_cast<uint32_t>(addressIndex), offset);
            }
        }
    }
    MCVR_PROFILE_METRIC("world_prepare.dense_entity_materials", denseMaterialCount);
    MCVR_PROFILE_METRIC("world_prepare.entity_materials", instanceOffsets.size());
    MCVR_PROFILE_METRIC("world_prepare.entity_material_bytes", instanceBytes.size());
    if (!instanceBytes.empty()) {
        ensureFrameBufferCapacity(entityGeometryInstances, vma, device, stagingBufferPool, frameResourceRetainer,
                                  instanceBytes.size(),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        for (const auto &[addressIndex, offset] : instanceOffsets)
            pbrBufferAddrs[addressIndex] = (entityGeometryInstances->bufferAddress() + offset) | 1ull;
        uploadVectorRange(entityGeometryInstances, cmdBuffer, instanceBytes);
    }

    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.uploadEntityRenderState");
        MCVR_PROFILE_METRIC("world_prepare.upload_entity_instance_bytes",
                            entityInstances.size() * sizeof(VkAccelerationStructureInstanceKHR));
        MCVR_PROFILE_METRIC("world_prepare.upload_entity_geometry_address_bytes",
                            indexBufferAddrs.size() * sizeof(uint64_t) * 4);
        MCVR_PROFILE_METRIC("world_prepare.upload_entity_last_transform_bytes",
                            lastObjToWorldMats.size() * sizeof(glm::mat4));
        uploadVectorRange(tlasInstanceBuffer, cmdBuffer, entityInstances);
        uploadVectorRange(blasOffsetsBuffer, cmdBuffer, blasOffsets);
        uploadVectorRange(indexBufferAddr, cmdBuffer, indexBufferAddrs);
        uploadVectorRange(pbrBufferAddr, cmdBuffer, pbrBufferAddrs);
        uploadVectorRange(lastIndexBufferAddr, cmdBuffer, lastIndexBufferAddrs);
        uploadVectorRange(lastPbrBufferAddr, cmdBuffer, lastPbrBufferAddrs);
        uploadVectorRange(lastObjToWorldMat, cmdBuffer, lastObjToWorldMats);
    }

    if (chunkGeometrySlotCount > 0) {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.copyChunkRenderState");
        const size_t chunkAddressBytes = static_cast<size_t>(chunkGeometrySlotCount) * sizeof(uint64_t);
        const size_t entityAddressBytes = static_cast<size_t>(entityGeometryCount) * sizeof(uint64_t);
        MCVR_PROFILE_METRIC("world_prepare.copy_chunk_address_bytes", chunkAddressBytes * 2);
        copyBufferRange(cmdBuffer, chunkMetadata.indexBufferAddressBuffer, indexBufferAddr, chunkAddressBytes, 0,
                        entityAddressBytes);
        copyBufferRange(cmdBuffer, chunkMetadata.pbrBufferAddressBuffer, pbrBufferAddr, chunkAddressBytes, 0,
                        entityAddressBytes);
    }

    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> rayTracingMetaData{{
        tlasInstanceBuffer,
        blasOffsetsBuffer,
        indexBufferAddr,
        pbrBufferAddr,
        lastIndexBufferAddr,
        lastPbrBufferAddr,
        lastObjToWorldMat,
        entityGeometryInstances,
    }};

    std::vector<vk::CommandBuffer::BufferMemoryBarrier> transferPostBarriers;
    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.buildTransferPostBarriers");
        transferPostBarriers.reserve(rayTracingMetaData.size());
        for (auto buffer : rayTracingMetaData) {
            if (buffer == nullptr) continue;
            transferPostBarriers.push_back({
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                .dstAccessMask =
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .buffer = buffer,
            });
        }
        MCVR_PROFILE_METRIC("world_prepare.transfer_post_barriers", transferPostBarriers.size());
    }
    cmdBuffer->barriersBufferImage(transferPostBarriers, {});

    if (chunkInstanceCount > 0) {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.dispatchChunkRenderState");
        const int32_t cameraBaseX = static_cast<int32_t>(std::floor(cameraPos.x));
        const int32_t cameraBaseY = static_cast<int32_t>(std::floor(cameraPos.y));
        const int32_t cameraBaseZ = static_cast<int32_t>(std::floor(cameraPos.z));
        glm::dvec3 lastCameraPos = cameraPos;
        auto lastWorldUniformBuffer = Renderer::instance().buffers()->lastWorldUniformBuffer();
        if (lastWorldUniformBuffer != nullptr && lastWorldUniformBuffer->mappedPtr() != nullptr) {
            const auto *lastWorldUbo = static_cast<const vk::Data::WorldUBO *>(lastWorldUniformBuffer->mappedPtr());
            lastCameraPos = glm::dvec3(lastWorldUbo->cameraPos.x, lastWorldUbo->cameraPos.y, lastWorldUbo->cameraPos.z);
        }
        const int32_t lastCameraBaseX = static_cast<int32_t>(std::floor(lastCameraPos.x));
        const int32_t lastCameraBaseY = static_cast<int32_t>(std::floor(lastCameraPos.y));
        const int32_t lastCameraBaseZ = static_cast<int32_t>(std::floor(lastCameraPos.z));
        ChunkRenderPushConstants pushConstants{
            .cameraBase = glm::ivec4(cameraBaseX, cameraBaseY, cameraBaseZ, 0),
            .cameraFraction = glm::vec4(static_cast<float>(cameraPos.x - static_cast<double>(cameraBaseX)),
                                        static_cast<float>(cameraPos.y - static_cast<double>(cameraBaseY)),
                                        static_cast<float>(cameraPos.z - static_cast<double>(cameraBaseZ)), 0.0f),
            .lastCameraBase = glm::ivec4(lastCameraBaseX, lastCameraBaseY, lastCameraBaseZ, 0),
            .lastCameraFraction =
                glm::vec4(static_cast<float>(lastCameraPos.x - static_cast<double>(lastCameraBaseX)),
                          static_cast<float>(lastCameraPos.y - static_cast<double>(lastCameraBaseY)),
                          static_cast<float>(lastCameraPos.z - static_cast<double>(lastCameraBaseZ)), 0.0f),
            .chunkCount = chunkInstanceCount,
            .entityInstanceCount = entityInstanceCount,
            .entityGeometryCount = entityGeometryCount,
            .entityHitGroupCount = entityHitGroupCount,
        };
        MCVR_PROFILE_METRIC("world_prepare.chunk_render_dispatch_groups", (chunkInstanceCount + 63) / 64);

        chunkRenderDescriptorTable->bindBuffer(chunkMetadata.renderRecordBuffer, 0, 0);
        chunkRenderDescriptorTable->bindBuffer(tlasInstanceBuffer, 0, 1);
        chunkRenderDescriptorTable->bindBuffer(blasOffsetsBuffer, 0, 2);
        chunkRenderDescriptorTable->bindBuffer(lastObjToWorldMat, 0, 3);
        chunkRenderDescriptorTable->bindBuffer(lastIndexBufferAddr, 0, 4);
        chunkRenderDescriptorTable->bindBuffer(lastPbrBufferAddr, 0, 5);
        chunkRenderDescriptorTable->bindBuffer(Renderer::instance().buffers()->worldUniformBuffer(), 0, 6);

        cmdBuffer->bindComputePipeline(chunkRenderPipeline)
            ->bindDescriptorTable(chunkRenderDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE);
        vkCmdPushConstants(cmdBuffer->vkCommandBuffer(), chunkRenderDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmdBuffer->vkCommandBuffer(), (chunkInstanceCount + 63) / 64, 1, 1);
    }

    std::vector<vk::CommandBuffer::BufferMemoryBarrier> readyBarriers;
    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.buildReadyBarriers");
        readyBarriers.reserve(rayTracingMetaData.size() + 3);
        for (auto buffer : rayTracingMetaData) {
            if (buffer == nullptr) continue;
            readyBarriers.push_back({
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .buffer = buffer,
            });
        }
        if (tlasInstanceBuffer != nullptr) {
            readyBarriers.push_back({
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .buffer = tlasInstanceBuffer,
            });
            readyBarriers.push_back({
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .buffer = tlasInstanceBuffer,
            });
        }
        if (lastObjToWorldMat != nullptr) {
            readyBarriers.push_back({
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .buffer = lastObjToWorldMat,
            });
        }
        MCVR_PROFILE_METRIC("world_prepare.ready_barriers", readyBarriers.size());
    }
    cmdBuffer->barriersBufferImage(readyBarriers, {});
}

void WorldPrepareContext::submitHighPriorityBLASBuilds() {
    if (highPriorityBLASBuildSubmitted) { return; }

    auto context = frameworkContext.lock();
    if (context == nullptr) { return; }

    auto world = Renderer::instance().world();
    if (world == nullptr) { return; }

    auto chunks = world->chunks();
    auto entities = world->entities();
    auto worldCommandBuffer = context->worldCommandBuffer;
    bool submitted = false;
    std::shared_ptr<std::vector<std::shared_ptr<vk::BLASBuilder>>> importantChunkBLASBuilders;

    {
        std::unique_lock<std::recursive_mutex> lock(chunks->mutex());
        importantChunkBLASBuilders = chunks->importantBLASBuilders();
    }

    if (importantChunkBLASBuilders != nullptr && !importantChunkBLASBuilders->empty()) {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.submitImportantChunkBLAS");
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "main_render/as/important_chunk_blas",
                                                       0.90f, 0.35f, 0.20f);
        vk::BLASBuilder::batchSubmit(*importantChunkBLASBuilders, worldCommandBuffer, importantChunkBLASBuilders);
        submitted = true;
    }

    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.submitEntityBLAS");
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "main_render/as/entity_blas", 0.90f, 0.40f,
                                                       0.25f);
        if (entities->submitPendingBLAS(worldCommandBuffer)) { submitted = true; }
    }

    if (submitted) {
        worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
            .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .srcAccessMask =
                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        }});
    }

    highPriorityBLASBuildSubmitted = true;
}

bool WorldPrepareContext::prepareAccelerationStructures() {
    if (accelerationStructuresReady) { return tlas != nullptr; }
    if (!accelerationStructureBuildPending || tlasBuilder == nullptr || tlasInstanceBuffer == nullptr ||
        pendingTlasInstanceCount == 0) {
        return tlas != nullptr;
    }

    MCVR_PROFILE_SCOPE("native.WorldPrepareContext.prepareAccelerationStructures");
    submitHighPriorityBLASBuilds();

    auto framework = Renderer::instance().framework();
    auto context = frameworkContext.lock();
    if (context == nullptr) { return false; }

    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto &frameResourceRetainer = framework->frameResourceRetainer();
    auto worldCommandBuffer = context->worldCommandBuffer;

    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.buildTLAS");
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "main_render/as/tlas", 0.90f, 0.45f, 0.30f);
        auto &instanceBuilder = tlasBuilder->beginInstanceBuilder();
        instanceBuilder.defineInstanceBuffer(tlasInstanceBuffer, pendingTlasInstanceCount);
        constexpr VkBuildAccelerationStructureFlagsKHR buildFlags =
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        const bool canRefit = builtTlasChunkRenderVersion == rasterChunkRenderVersion && tlasRefitCount < 32 &&
                              tlasBuilder->canUpdate(buildFlags, reusableTlas);
        if (canRefit) {
            tlasBuilder->defineUpdateProperty(buildFlags, reusableTlas);
            worldCommandBuffer->barriersMemory({{
                .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                .srcAccessMask =
                    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                .dstAccessMask =
                    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            }});
        } else {
            tlasBuilder->defineBuildProperty(buildFlags);
        }
        tlasBuilder->querySizeInfo(device);
        const VkDeviceSize requiredTlasStorageBytes = tlasBuilder->requiredStorageSize();
        const VkDeviceSize requiredTlasScratchBytes = tlasBuilder->requiredScratchSize();
        if (tlasStorageBuffer == nullptr || tlasStorageBuffer->size() < requiredTlasStorageBytes) {
            frameResourceRetainer.retain(tlasStorageBuffer);
            tlasStorageBuffer = vk::DeviceLocalBuffer::create(vma, device, false, requiredTlasStorageBytes,
                                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                              0, VMA_MEMORY_USAGE_GPU_ONLY, 256);
        }
        const VkDeviceSize scratchAlignment =
            physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment;
        if (tlasScratchBuffer == nullptr || tlasScratchBuffer->size() < requiredTlasScratchBytes) {
            frameResourceRetainer.retain(tlasScratchBuffer);
            tlasScratchBuffer = vk::DeviceLocalBuffer::create(vma, device, false, requiredTlasScratchBytes,
                                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                              0, VMA_MEMORY_USAGE_GPU_ONLY, scratchAlignment);
        }
        tlas = tlasBuilder->useExternalStorage(tlasStorageBuffer, 0, requiredTlasStorageBytes)
                   ->useExternalScratch(tlasScratchBuffer)
                   ->buildAndSubmit(device, worldCommandBuffer);
        reusableTlas = tlas;
        builtTlasChunkRenderVersion = rasterChunkRenderVersion;
        tlasRefitCount = canRefit ? tlasRefitCount + 1 : 0;
    }

    worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    }});

    accelerationStructureBuildPending = false;
    accelerationStructuresReady = tlas != nullptr;
    return accelerationStructuresReady;
}

void WorldPrepareContext::flushPendingHighPriorityBLASBuilds() {
    if (!highPriorityBLASBuildSubmitted) { submitHighPriorityBLASBuilds(); }
}

void WorldPrepareContext::render(ShaderPackLoader::AccelerationStructureBuildMode accelerationStructureBuildMode) {
    MCVR_PROFILE_SCOPE("native.WorldPrepareContext.render");
    std::shared_ptr<MainRenderModuleContext> mainRenderContext;
    std::shared_ptr<MainRenderModule> mainRenderModule;
    std::shared_ptr<WorldPrepare> worldPrepare1;
    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.resolveRenderContext");
        mainRenderContext = mainRenderModuleContext.lock();
        mainRenderModule = mainRenderContext != nullptr ? mainRenderContext->mainRenderModule.lock() : nullptr;
        worldPrepare1 = worldPrepare.lock();
    }
    if (mainRenderModule == nullptr) { return; }
    if (worldPrepare1 == nullptr) { return; }

    auto framework = Renderer::instance().framework();
    if (framework == nullptr) { return; }

    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.resetFrameState");
        framework->frameResourceRetainer().retainMany(tlas, tlasBuilder);
        tlas = nullptr;
        tlasBuilder = nullptr;
        pendingTlasInstanceCount = 0;
        highPriorityBLASBuildSubmitted = false;
        accelerationStructureBuildPending = false;
        accelerationStructuresReady = false;
    }

    std::shared_ptr<FrameworkContext> context;
    std::shared_ptr<vk::CommandBuffer> worldCommandBuffer;
    decltype(Renderer::instance().world()->chunks()) chunks;
    decltype(Renderer::instance().world()->entities()) entities;
    glm::dvec3 cameraPos;
    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.acquireWorldState");
        context = frameworkContext.lock();
        if (context == nullptr) { return; }
        worldCommandBuffer = context->worldCommandBuffer;
        worldCommandBuffer->retainResource(shared_from_this());

        auto world = Renderer::instance().world();
        chunks = world->chunks();
        entities = world->entities();
        cameraPos = world->getCameraPos();
    }

    auto chunkBuildScheduler = chunks->chunkBuildScheduler();
    if (chunkBuildScheduler != nullptr) {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.processChunkBuildScheduler");
        {
            MCVR_PROFILE_SCOPE("native.WorldPrepareContext.checkChunkBuildBatches");
            chunkBuildScheduler->tryCheckBatchesFinish();
        }
        {
            MCVR_PROFILE_SCOPE("native.WorldPrepareContext.scheduleChunkBuildBatches");
            chunkBuildScheduler->tryScheduleBatches(chunkBuildScheduler->chunkBuildingBatchSize());
        }
    }

    if (accelerationStructureBuildMode == ShaderPackLoader::AccelerationStructureBuildMode::Sync) {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.syncHighPriorityBLAS");
        submitHighPriorityBLASBuilds();
    }

    uint32_t blasAccu = 0, blasGroupAccu = 0;
    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.clearScratchState");
        entityHitGroupNames.clear();
        scratchBlasOffsets.clear();
        scratchIndexBufferAddrs.clear();
        scratchPbrBufferAddrs.clear();
        scratchLastIndexBufferAddrs.clear();
        scratchLastPbrBufferAddrs.clear();
        scratchLastObjToWorldMats.clear();
        chunkHitGroupDirtyRanges.clear();
    }

    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.createTlasBuilder");
        tlasBuilder = vk::TLASBuilder::create();
    }
    auto &instanceBuilder = tlasBuilder->beginInstanceBuilder();
    int blasIndex = 0;
    ChunkGeometryMetadataView chunkGeometryMetadataSnapshot;
    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.snapshotChunkMetadata");
        std::unique_lock<std::recursive_mutex> lock(chunks->mutex());
        chunks->uploadChunkGeometryMetadataCache(worldCommandBuffer);
        auto worldUniformBuffer = Renderer::instance().buffers()->worldUniformBuffer();
        auto worldUniforms = static_cast<vk::Data::WorldUBO *>(worldUniformBuffer->mappedPtr());
        worldUniforms->chunkGridInfo = chunks->chunkGridInfo();
        worldUniforms->chunkStorageSectionPos = chunks->chunkStorageSectionPos();
        const auto &chunkGeometryMetadataCache = chunks->chunkGeometryMetadataCache();
        if (worldUniforms->vistaDistanceBlocks != 0) {
            Renderer::instance().world()->vista()->updateCoverage(
                chunks->chunks(), chunkGeometryMetadataCache.renderRecords,
                worldUniforms->chunkGridInfo, worldUniforms->chunkStorageSectionPos,
                chunkGeometryMetadataCache.renderVersion);
        }
        const auto &liveRenderableChunks = chunks->renderableChunks();
        const auto &liveRenderableChunkSnapshots = chunks->renderableChunkSnapshots();
        MCVR_PROFILE_METRIC("world_prepare.live_renderable_chunks", liveRenderableChunks.size());
        MCVR_PROFILE_METRIC("world_prepare.live_renderable_chunk_snapshots", liveRenderableChunkSnapshots.size());
        MCVR_PROFILE_METRIC("world_prepare.render_records", chunkGeometryMetadataCache.renderRecords.size());
        MCVR_PROFILE_METRIC("world_prepare.raster_dirty_slots", chunkGeometryMetadataCache.rasterDrawDirtySlots.size());
        MCVR_PROFILE_METRIC("world_prepare.render_record_dirty_ranges",
                            chunkGeometryMetadataCache.renderRecordDirtyRanges.size());
        MCVR_PROFILE_METRIC("world_prepare.geometry_dirty_ranges",
                            chunkGeometryMetadataCache.geometryDirtyRanges.size());
        MCVR_PROFILE_METRIC("world_prepare.hit_group_dirty_ranges",
                            chunkGeometryMetadataCache.hitGroupDirtyRanges.size());
        {
            MCVR_PROFILE_SCOPE("native.WorldPrepareContext.copyRenderableChunkRefs");
            rasterRenderableChunks = liveRenderableChunkSnapshots;
        }
        {
            MCVR_PROFILE_SCOPE("native.WorldPrepareContext.copyRasterChunkMetadata");
            rasterChunkRenderRecords = chunkGeometryMetadataCache.renderRecords;
            rasterChunkDrawDirtySlots = chunkGeometryMetadataCache.rasterDrawDirtySlots;
            rasterChunkGeometrySlotCount = static_cast<uint32_t>(chunkGeometryMetadataCache.geometrySlotCount);
            rasterChunkHitGroupVersion = chunkGeometryMetadataCache.hitGroupVersion;
            rasterChunkRenderVersion = chunkGeometryMetadataCache.renderVersion;
            rasterChunkDrawDirtyStartVersion = chunkGeometryMetadataCache.rasterDrawDirtyStartVersion;
        }
        if (!hasValidChunkHitGroupNames || chunkHitGroupVersion != chunkGeometryMetadataCache.hitGroupVersion) {
            MCVR_PROFILE_SCOPE("native.WorldPrepareContext.updateChunkHitGroups");
            MCVR_PROFILE_METRIC("world_prepare.chunk_hit_group_names", chunkGeometryMetadataCache.hitGroupNames.size());
            if (hasValidChunkHitGroupNames &&
                chunkHitGroupNames.size() == chunkGeometryMetadataCache.hitGroupNames.size() &&
                !chunkGeometryMetadataCache.hitGroupDirtyRanges.empty()) {
                copyDirtyHitGroupNameRanges(chunkHitGroupNames, chunkGeometryMetadataCache.hitGroupNames,
                                            chunkGeometryMetadataCache.hitGroupDirtyRanges, chunkHitGroupDirtyRanges);
            } else {
                chunkHitGroupNames = chunkGeometryMetadataCache.hitGroupNames;
                mergeHitGroupRecordRange(chunkHitGroupDirtyRanges, 0, static_cast<uint32_t>(chunkHitGroupNames.size()));
            }
            chunkHitGroupVersion = chunkGeometryMetadataCache.hitGroupVersion;
            hasValidChunkHitGroupNames = true;
        }

        if (!chunkGeometryMetadataCache.hitGroupDirtyRanges.empty()) {
            MCVR_PROFILE_SCOPE("native.WorldPrepareContext.maybeClearHitGroupDirtyRanges");
            bool areAllContextsUpdated = true;
            for (const auto &context : worldPrepare1->contexts_) {
                if (context == nullptr) { continue; }
                if (!context->hasValidChunkHitGroupNames ||
                    context->chunkHitGroupVersion != chunkGeometryMetadataCache.hitGroupVersion) {
                    areAllContextsUpdated = false;
                    break;
                }
            }
            if (areAllContextsUpdated) { chunks->clearChunkGeometryHitGroupDirtyRanges(); }
        }

        if (!chunkGeometryMetadataCache.rasterDrawDirtySlots.empty()) {
            MCVR_PROFILE_SCOPE("native.WorldPrepareContext.maybeClearRasterDirtySlots");
            bool areAllContextsUpdated = true;
            for (const auto &context : worldPrepare1->contexts_) {
                if (context == nullptr) { continue; }
                if (context->rasterChunkRenderVersion != chunkGeometryMetadataCache.renderVersion) {
                    areAllContextsUpdated = false;
                    break;
                }
            }
            if (areAllContextsUpdated) { chunks->clearChunkRasterDrawDirtySlots(); }
        }

        {
            MCVR_PROFILE_SCOPE("native.WorldPrepareContext.copyChunkMetadataSnapshot");
            const size_t snapshotBytes =
                chunkGeometryMetadataCache.renderRecords.size() * sizeof(ChunkRenderRecord) +
                chunkGeometryMetadataCache.indexBufferAddresses.size() * sizeof(VkDeviceAddress) +
                chunkGeometryMetadataCache.pbrBufferAddresses.size() * sizeof(VkDeviceAddress) +
                chunkGeometryMetadataCache.hitGroupNames.size() * sizeof(const std::string *) +
                chunkGeometryMetadataCache.chunkAllocations.size() * sizeof(ChunkMetadataSlabAllocationInfo) +
                chunkGeometryMetadataCache.renderRecordDirtyRanges.size() * sizeof(ChunkMetadataUpdateRangeInfo) +
                chunkGeometryMetadataCache.geometryDirtyRanges.size() * sizeof(ChunkMetadataUpdateRangeInfo) +
                chunkGeometryMetadataCache.hitGroupDirtyRanges.size() * sizeof(ChunkMetadataUpdateRangeInfo) +
                chunkGeometryMetadataCache.rasterDrawDirtySlots.size() * sizeof(uint32_t);
            MCVR_PROFILE_METRIC("world_prepare.chunk_metadata_snapshot_bytes", snapshotBytes);
            chunkGeometryMetadataSnapshot = {chunkGeometryMetadataCache.renderRecords.size(),
                                             chunkGeometryMetadataCache.geometrySlotCount,
                                             chunkGeometryMetadataCache.actualGeometryCount,
                                             chunkGeometryMetadataCache.renderRecordBuffer,
                                             chunkGeometryMetadataCache.indexBufferAddressBuffer,
                                             chunkGeometryMetadataCache.pbrBufferAddressBuffer};
        }
    }
    const auto &chunkGeometryMetadataCache = chunkGeometryMetadataSnapshot;
    const size_t chunkInstanceCount = chunkGeometryMetadataCache.renderRecordCount;
    const size_t chunkGeometrySlotCount = chunkGeometryMetadataCache.geometrySlotCount;
    const size_t chunkGeometryCount = chunkGeometryMetadataCache.actualGeometryCount;

    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.collectEntities");
        auto entityBatch = entities->entityBatch();
        rasterEntityBatch = entityBatch;
        const size_t entityCount = entityBatch != nullptr ? entityBatch->entities.size() : 0;
        MCVR_PROFILE_METRIC("world_prepare.entity_count", entityCount);

        if (entityBatch != nullptr) {
            size_t entityGeometryCount = 0;
            {
                MCVR_PROFILE_SCOPE("native.WorldPrepareContext.countEntityGeometry");
                for (const auto &entity : entityBatch->entities) { entityGeometryCount += entity->geometryCount; }
            }
            MCVR_PROFILE_METRIC("world_prepare.entity_geometry_count", entityGeometryCount);
            {
                MCVR_PROFILE_SCOPE("native.WorldPrepareContext.reserveEntityScratch");
                instanceBuilder.instances.reserve(entityBatch->entities.size());
                scratchBlasOffsets.reserve(entityBatch->entities.size());
                entityHitGroupNames.reserve(entityBatch->entities.size() + entityGeometryCount);
                scratchIndexBufferAddrs.reserve(entityGeometryCount);
                scratchPbrBufferAddrs.reserve(entityGeometryCount);
                scratchLastIndexBufferAddrs.reserve(entityGeometryCount);
                scratchLastPbrBufferAddrs.reserve(entityGeometryCount);
                scratchLastObjToWorldMats.reserve(entityBatch->entities.size());
            }

            auto worldUniformBuffer = Renderer::instance().buffers()->worldUniformBuffer();
            auto worldUniforms = static_cast<vk::Data::WorldUBO *>(worldUniformBuffer->mappedPtr());

            auto &renderEntities = entityBatch->entities;
            {
                MCVR_PROFILE_SCOPE("native.WorldPrepareContext.buildEntityInstances");
                for (int i = 0; i < renderEntities.size(); i++) {
                    VkGeometryInstanceFlagsKHR flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                    VkTransformMatrixKHR transform;

                    if (renderEntities[i]->prebuiltBLAS < 0) {
                        if (renderEntities[i]->coordinate == World::Coordinates::WORLD || !worldUniforms) {
                            transform = {
                                1, 0, 0, static_cast<float>(renderEntities[i]->x - cameraPos.x),
                                0, 1, 0, static_cast<float>(renderEntities[i]->y - cameraPos.y),
                                0, 0, 1, static_cast<float>(renderEntities[i]->z - cameraPos.z),
                            };
                        } else if (renderEntities[i]->coordinate == World::Coordinates::CAMERA) {
                            glm::mat4 viewMat = glm::transpose(worldUniforms->cameraViewMatInv);

                            transform = {
                                viewMat[0][0], viewMat[0][1], viewMat[0][2], viewMat[0][3],
                                viewMat[1][0], viewMat[1][1], viewMat[1][2], viewMat[1][3],
                                viewMat[2][0], viewMat[2][1], viewMat[2][2], viewMat[2][3],
                            };
                        } else if (renderEntities[i]->coordinate == World::Coordinates::CAMERA_SHIFT) {
                            glm::vec3 shift = glm::vec3(worldUniforms->cameraViewMatInv[3]);
                            transform = {
                                1, 0, 0, shift.x, 0, 1, 0, shift.y, 0, 0, 1, shift.z,
                            };
                        }

                        instanceBuilder.defineInstance(transform, blasIndex, renderEntities[i]->rayTracingFlag,
                                                       blasGroupAccu, flags, renderEntities[i]->blas);
                    } else {
                        glm::mat4 model = renderEntities[i]->localTransform;
                        if (glm::determinant(glm::mat3(model)) < 0) flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;
                        model[3] += glm::vec4(static_cast<float>(renderEntities[i]->x - cameraPos.x),
                                              static_cast<float>(renderEntities[i]->y - cameraPos.y),
                                              static_cast<float>(renderEntities[i]->z - cameraPos.z), 0.0f);
                        for (int row = 0; row < 3; ++row) {
                            for (int column = 0; column < 4; ++column) { transform.matrix[row][column] = model[column][row]; }
                        }
                        instanceBuilder.defineInstance(transform, blasIndex, renderEntities[i]->rayTracingFlag,
                                                       blasGroupAccu, flags, renderEntities[i]->blas);
                    }

                    entityHitGroupNames.push_back(&shadowHitGroupName);
                    for (int j = 0; j < renderEntities[i]->geometryCount; j++) {
                        const std::string *groupName =
                            renderEntities[i]->geometryGroupNames != nullptr &&
                                    j < static_cast<int>(renderEntities[i]->geometryGroupNames->size()) ?
                                &(*renderEntities[i]->geometryGroupNames)[j] :
                                &defaultHitGroupName;
                        entityHitGroupNames.push_back(groupName);
                    }

                    for (int j = 0; j < renderEntities[i]->geometryCount; j++) {
                        scratchIndexBufferAddrs.push_back((*renderEntities[i]->indexBufferAddresses)[j]);
                        scratchPbrBufferAddrs.push_back((*renderEntities[i]->pbrBufferAddresses)[j]);
                    }

                    {
                        auto transformToMat4 = [](const VkTransformMatrixKHR &matrix) {
                            return glm::transpose(
                                glm::mat4(glm::make_vec4(matrix.matrix[0]), glm::make_vec4(matrix.matrix[1]),
                                          glm::make_vec4(matrix.matrix[2]), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)));
                        };
                        glm::mat4 lastObjToWorldMat = transformToMat4(transform);
                        if (renderEntities[i]->hasSavedRenderState) {
                            bool hasAnyCompatiblePreviousGeometry = false;
                            const size_t previousGeometryCount = renderEntities[i]->lastFrameIndexCounts.size();
                            if (renderEntities[i]->geometryGroupNames != nullptr &&
                                renderEntities[i]->geometryContentNames != nullptr &&
                                renderEntities[i]->geometryGroupNames->size() == renderEntities[i]->geometryCount &&
                                renderEntities[i]->geometryContentNames->size() == renderEntities[i]->geometryCount &&
                                renderEntities[i]->lastFrameVertexCounts.size() == previousGeometryCount &&
                                renderEntities[i]->lastFrameIndexBufferAddresses.size() == previousGeometryCount &&
                                renderEntities[i]->lastFramePbrBufferAddresses.size() == previousGeometryCount &&
                                renderEntities[i]->lastFrameGeometryGroupNames.size() == previousGeometryCount &&
                                renderEntities[i]->lastFrameGeometryContentNames.size() == previousGeometryCount) {
                                std::vector<bool> previousGeometryMatched(previousGeometryCount, false);
                                for (int j = 0; j < renderEntities[i]->geometryCount; j++) {
                                    size_t matchedGeometry = previousGeometryCount;
                                    for (size_t previousIndex = 0; previousIndex < previousGeometryCount;
                                         previousIndex++) {
                                        if (!previousGeometryMatched[previousIndex] &&
                                            renderEntities[i]->lastFrameGeometryGroupNames[previousIndex] ==
                                                (*renderEntities[i]->geometryGroupNames)[j] &&
                                            renderEntities[i]->lastFrameGeometryContentNames[previousIndex] ==
                                                (*renderEntities[i]->geometryContentNames)[j] &&
                                            renderEntities[i]->lastFrameVertexCounts[previousIndex] ==
                                                (*renderEntities[i]->vertexCounts)[j] &&
                                            renderEntities[i]->lastFrameIndexCounts[previousIndex] ==
                                                (*renderEntities[i]->indexCounts)[j]) {
                                            matchedGeometry = previousIndex;
                                            break;
                                        }
                                    }

                                    if (matchedGeometry < previousGeometryCount) {
                                        previousGeometryMatched[matchedGeometry] = true;
                                        hasAnyCompatiblePreviousGeometry = true;
                                        scratchLastIndexBufferAddrs.push_back(
                                            renderEntities[i]->lastFrameIndexBufferAddresses[matchedGeometry]);
                                        scratchLastPbrBufferAddrs.push_back(
                                            renderEntities[i]->lastFramePbrBufferAddresses[matchedGeometry]);
                                    } else {
                                        scratchLastIndexBufferAddrs.push_back(0);
                                        scratchLastPbrBufferAddrs.push_back(0);
                                    }
                                }
                            } else {
                                for (int j = 0; j < renderEntities[i]->geometryCount; j++) {
                                    scratchLastIndexBufferAddrs.push_back(0);
                                    scratchLastPbrBufferAddrs.push_back(0);
                                }
                            }

                            if (hasAnyCompatiblePreviousGeometry) {
                                lastObjToWorldMat = transformToMat4(renderEntities[i]->lastFrameTransform);
                            }
                        } else {
                            for (int j = 0; j < renderEntities[i]->geometryCount; j++) {
                                scratchLastIndexBufferAddrs.push_back(0);
                                scratchLastPbrBufferAddrs.push_back(0);
                            }
                        }
                        scratchLastObjToWorldMats.push_back(lastObjToWorldMat);
                    }

                    renderEntities[i]->saveRenderStateForNextFrame(transform);

                    scratchBlasOffsets.push_back(blasAccu);
                    blasAccu += renderEntities[i]->geometryCount;
                    blasGroupAccu += renderEntities[i]->geometryCount + 1;

                    blasIndex++;
                }
            }
        }
    }

    auto vistaSnapshots = Renderer::instance().world()->vista()->prepare(worldCommandBuffer);
    vistaInstanceCount = static_cast<uint32_t>(vistaSnapshots.size());
    auto worldUniformBuffer = Renderer::instance().buffers()->worldUniformBuffer();
    auto worldUniforms = static_cast<vk::Data::WorldUBO *>(worldUniformBuffer->mappedPtr());
    worldUniforms->vistaInstanceOffset = blasIndex;
    worldUniforms->vistaInstanceCount = vistaInstanceCount;
    const VkDeviceAddress coverage = worldUniforms->vistaDistanceBlocks == 0 ? 0 :
        Renderer::instance().world()->vista()->prepareCoverage(worldCommandBuffer);
    const auto firstBorder = std::find_if(vistaSnapshots.begin(), vistaSnapshots.end(),
                                         [](const auto &instance) { return instance.isBorder; });
    const uint32_t borderOffset = static_cast<uint32_t>(firstBorder - vistaSnapshots.begin());
    worldUniforms->vistaCoverage =
        glm::ivec4(static_cast<uint32_t>(coverage), static_cast<uint32_t>(coverage >> 32),
                   blasIndex + borderOffset, vistaInstanceCount - borderOffset);
    worldUniformBuffer->flush();
    for (const auto &instance : vistaSnapshots) {
        const auto &data = instance.data;
        const glm::dvec3 position = glm::dvec3(data->x, data->y, data->z) + data->offset;
        VkTransformMatrixKHR transform = {
            1, 0, 0, static_cast<float>(position.x - cameraPos.x),
            0, 1, 0, static_cast<float>(position.y - cameraPos.y),
            0, 0, 1, static_cast<float>(position.z - cameraPos.z),
        };
        instanceBuilder.defineInstance(transform, blasIndex++, 0x01, blasGroupAccu,
            VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR |
                (instance.isOpaque ? VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR : 0), data->blas);
        entityHitGroupNames.push_back(&shadowHitGroupName);
        entityHitGroupNames.push_back(&defaultHitGroupName);
        scratchBlasOffsets.push_back(blasAccu | 0x80000000u);
        scratchIndexBufferAddrs.push_back(data->indexBuffer->bufferAddress());
        scratchPbrBufferAddrs.push_back(data->pbrBuffer->bufferAddress());
        scratchLastIndexBufferAddrs.push_back(instance.hasHistory ? data->indexBuffer->bufferAddress() : 0);
        scratchLastPbrBufferAddrs.push_back(instance.hasHistory ? data->pbrBuffer->bufferAddress() : 0);
        const auto currentTransform = glm::translate(glm::mat4(1.0f), glm::vec3(position - cameraPos));
        scratchLastObjToWorldMats.push_back(instance.hasHistory ? data->previousTransform : currentTransform);
        data->previousTransform = currentTransform;
        ++blasAccu;
        blasGroupAccu += 2;
    }

    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.indexEntityHitGroupNames");
        std::unordered_map<std::string_view, uint32_t> names;
        uniqueEntityHitGroupNames.clear();
        entityHitGroupNameIds.clear();
        entityHitGroupNameIds.reserve(entityHitGroupNames.size());
        for (const auto *name : entityHitGroupNames) {
            const auto [it, inserted] = names.try_emplace(*name, uniqueEntityHitGroupNames.size());
            if (inserted) uniqueEntityHitGroupNames.push_back(name);
            entityHitGroupNameIds.push_back(it->second);
        }
        MCVR_PROFILE_METRIC("world_prepare.unique_entity_hit_group_names", uniqueEntityHitGroupNames.size());
    }
    const uint32_t entityHitGroupCount = blasGroupAccu;
    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.collectChunks");
        MCVR_PROFILE_METRIC("renderable_chunks", chunkInstanceCount);
        MCVR_PROFILE_METRIC("chunk_geometries", chunkGeometryCount);
        MCVR_PROFILE_METRIC("chunk_geometry_slots", chunkGeometrySlotCount);
        MCVR_PROFILE_METRIC("chunk_hit_groups", chunkHitGroupNames.size());
    }

    const uint32_t totalInstanceCount = static_cast<uint32_t>(instanceBuilder.instances.size() + chunkInstanceCount);
    if (totalInstanceCount == 0) {
        tlas = nullptr;
        tlasBuilder = nullptr;
        framework->frameResourceRetainer().retain(reusableTlas);
        reusableTlas = nullptr;
        tlasRefitCount = 0;
        accelerationStructureBuildPending = false;
        accelerationStructuresReady = false;
        return;
    }

    std::vector<VkAccelerationStructureInstanceKHR> entityInstances;
    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.buildEntityInstanceVector");
        entityInstances = instanceBuilder.buildInstances();
        MCVR_PROFILE_METRIC("world_prepare.entity_instance_vector_count", entityInstances.size());
    }
    prepareRayTracingBuffers(entityInstances, scratchBlasOffsets, scratchIndexBufferAddrs, scratchPbrBufferAddrs,
                             scratchLastIndexBufferAddrs, scratchLastPbrBufferAddrs, scratchLastObjToWorldMats,
                             chunkGeometryMetadataCache, entityHitGroupCount, cameraPos);

    {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.markAccelerationStructuresPending");
        pendingTlasInstanceCount = totalInstanceCount;
        accelerationStructureBuildPending = true;
        accelerationStructuresReady = false;
    }

    if (accelerationStructureBuildMode == ShaderPackLoader::AccelerationStructureBuildMode::Sync) {
        MCVR_PROFILE_SCOPE("native.WorldPrepareContext.syncPrepareAccelerationStructures");
        prepareAccelerationStructures();
    }
}

void WorldPrepareContext::setupHitGroupSbt(const std::unordered_map<std::string, uint32_t> &hitGroupNameToIndex,
                                           uint32_t fallbackHitGroupIndex,
                                           uint32_t shadowHitGroupIndex,
                                           std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                           std::shared_ptr<vk::SBT> updateSbt,
                                           std::shared_ptr<vk::SBT> querySbt) {
    MCVR_PROFILE_SCOPE("native.WorldPrepareContext.setupHitGroupSbt");
    uint64_t dirtyRecordCount = 0;
    for (const auto &range : chunkHitGroupDirtyRanges) { dirtyRecordCount += range.count; }
    MCVR_PROFILE_METRIC("world_prepare.setup_sbt_chunk_hit_group_count", chunkHitGroupNames.size());
    MCVR_PROFILE_METRIC("world_prepare.setup_sbt_dirty_range_count", chunkHitGroupDirtyRanges.size());
    MCVR_PROFILE_METRIC("world_prepare.setup_sbt_dirty_record_count", dirtyRecordCount);
    resolveHitGroupIndices(uniqueEntityHitGroupNames, hitGroupNameToIndex, fallbackHitGroupIndex, shadowHitGroupIndex,
                           scratchUniqueEntityHitGroupIndices);
    scratchEntityHitGroupIndices.resize(entityHitGroupNameIds.size());
    for (size_t i = 0; i < entityHitGroupNameIds.size(); ++i)
        scratchEntityHitGroupIndices[i] = scratchUniqueEntityHitGroupIndices[entityHitGroupNameIds[i]];

    auto setupSbt = [&](const std::shared_ptr<vk::SBT> &sbt) {
        if (sbt == nullptr) { return; }

        ChunkHitGroupIndexInfo &indexInfo = chunkHitGroupIndexInfoBySbt[sbt.get()];
        const bool isMappingDirty = indexInfo.hitGroupNameMap != &hitGroupNameToIndex ||
                                    indexInfo.fallbackHitGroupIndex != fallbackHitGroupIndex ||
                                    indexInfo.shadowHitGroupIndex != shadowHitGroupIndex;
        const bool isLayoutDirty = indexInfo.hitGroupIndices.size() != chunkHitGroupNames.size();
        const bool isVersionDirty = indexInfo.chunkHitGroupVersion != chunkHitGroupVersion;
        if (isMappingDirty || isLayoutDirty || (isVersionDirty && chunkHitGroupDirtyRanges.empty())) {
            resolveHitGroupIndices(chunkHitGroupNames, hitGroupNameToIndex, fallbackHitGroupIndex, shadowHitGroupIndex,
                                   indexInfo.hitGroupIndices);
            indexInfo.dirtyHitGroupRecordRanges.clear();
            mergeHitGroupRecordRange(indexInfo.dirtyHitGroupRecordRanges, 0,
                                     static_cast<uint32_t>(indexInfo.hitGroupIndices.size()));
            indexInfo.sbtSuffixVersion = ++chunkHitGroupIndexVersionCounter;
        } else if (isVersionDirty) {
            resolveDirtyHitGroupIndexRanges(chunkHitGroupNames, hitGroupNameToIndex, fallbackHitGroupIndex,
                                            shadowHitGroupIndex, chunkHitGroupDirtyRanges, indexInfo.hitGroupIndices,
                                            indexInfo.dirtyHitGroupRecordRanges);
            indexInfo.sbtSuffixVersion = ++chunkHitGroupIndexVersionCounter;
        } else {
            indexInfo.dirtyHitGroupRecordRanges.clear();
        }

        if (isMappingDirty || isLayoutDirty || isVersionDirty) {
            indexInfo.hitGroupNameMap = &hitGroupNameToIndex;
            indexInfo.fallbackHitGroupIndex = fallbackHitGroupIndex;
            indexInfo.shadowHitGroupIndex = shadowHitGroupIndex;
            indexInfo.chunkHitGroupVersion = chunkHitGroupVersion;
        }

        sbt->setupHitSBT(scratchEntityHitGroupIndices, indexInfo.hitGroupIndices, indexInfo.dirtyHitGroupRecordRanges,
                         indexInfo.sbtSuffixVersion, commandBuffer);
    };

    setupSbt(updateSbt);
    setupSbt(querySbt);
}
