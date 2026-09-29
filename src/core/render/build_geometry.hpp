#pragma once

#include "common/shared.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <cstdint>
#include <memory>
#include <vector>

struct BuildBufferLease {
    virtual ~BuildBufferLease() = default;

    virtual bool empty() const = 0;
    virtual bool valid(uint32_t index) const = 0;
    virtual std::shared_ptr<void> retain(uint32_t index) const = 0;
};

struct BuildGeometry {
    enum class VertexSource {
        ExternalPBR,
        MaterializedPBR,
    };

    enum class IndexSource {
        GeneratedQuads,
        ExternalU16,
        ExternalU32,
        MaterializedU32,
    };

    VertexSource vertexSource = VertexSource::MaterializedPBR;
    IndexSource indexSource = IndexSource::GeneratedQuads;
    const vk::VertexFormat::EntityPBRVertex *externalVertices = nullptr;
    const void *externalIndices = nullptr;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    bool inputReleased = false;
    std::shared_ptr<void> lifetime;
    std::vector<vk::VertexFormat::EntityPBRVertex> materializedVertices;
    std::vector<uint32_t> materializedIndices;

    static BuildGeometry externalPBR(const vk::VertexFormat::EntityPBRVertex *vertices,
                                     uint32_t vertexCount,
                                     IndexSource indexSource,
                                     const void *indices,
                                     uint32_t indexCount,
                                     std::shared_ptr<void> lifetime = nullptr);

    static BuildGeometry materializedPBR(std::vector<vk::VertexFormat::EntityPBRVertex> &&vertices,
                                         std::vector<uint32_t> &&indices);

    static uint32_t generatedQuadIndex(uint32_t index);

    const vk::VertexFormat::EntityPBRVertex *vertexData() const;
    const vk::VertexFormat::EntityPBRVertex &vertex(uint32_t index) const;
    uint32_t index(uint32_t index) const;
    bool empty() const;
    void releaseInput();
};

struct ChunkBuildGeometry {
    enum class VertexSource {
        ExternalPBR,
        MaterializedPBR,
    };

    enum class IndexSource {
        GeneratedQuads,
        ExternalU16,
        ExternalU32,
        MaterializedU32,
    };

    VertexSource vertexSource = VertexSource::MaterializedPBR;
    IndexSource indexSource = IndexSource::GeneratedQuads;
    const vk::VertexFormat::ChunkPBRVertex *externalVertices = nullptr;
    const void *externalIndices = nullptr;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    bool inputReleased = false;
    std::shared_ptr<void> lifetime;
    std::vector<vk::VertexFormat::ChunkPBRVertex> materializedVertices;
    std::vector<uint32_t> materializedIndices;

    static ChunkBuildGeometry externalPBR(const vk::VertexFormat::ChunkPBRVertex *vertices,
                                          uint32_t vertexCount,
                                          IndexSource indexSource,
                                          const void *indices,
                                          uint32_t indexCount,
                                          std::shared_ptr<void> lifetime = nullptr);

    static ChunkBuildGeometry materializedPBR(std::vector<vk::VertexFormat::ChunkPBRVertex> &&vertices,
                                              std::vector<uint32_t> &&indices);

    static uint32_t generatedQuadIndex(uint32_t index);

    const vk::VertexFormat::ChunkPBRVertex *vertexData() const;
    const vk::VertexFormat::ChunkPBRVertex &vertex(uint32_t index) const;
    uint32_t index(uint32_t index) const;
    bool empty() const;
    void releaseInput();
};

void releaseBuildGeometryInputs(std::vector<BuildGeometry> &geometries);
void releaseBuildGeometryInputs(std::vector<ChunkBuildGeometry> &geometries);

void buildPackedGeometry(const std::vector<BuildGeometry> &geometries,
                         std::vector<vk::VertexFormat::EntityPBRVertex> &packedVertices,
                         std::vector<uint32_t> &packedIndices);
void buildPackedGeometry(const std::vector<ChunkBuildGeometry> &geometries,
                         std::vector<vk::VertexFormat::ChunkPBRVertex> &packedVertices,
                         std::vector<uint32_t> &packedIndices);
