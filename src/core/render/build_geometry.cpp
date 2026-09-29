#include "core/render/build_geometry.hpp"

#include "core/vulkan/vertex.hpp"

#include <stdexcept>

static_assert(sizeof(vk::VertexFormat::EntityPBRVertex) == 64);
static_assert(sizeof(vk::VertexFormat::ChunkPBRVertex) == 40);

uint32_t BuildGeometry::generatedQuadIndex(uint32_t index) {
    static constexpr uint32_t pattern[6] = {0, 1, 2, 2, 3, 0};
    return (index / 6) * 4 + pattern[index % 6];
}

BuildGeometry BuildGeometry::externalPBR(const vk::VertexFormat::EntityPBRVertex *vertices,
                                         uint32_t vertexCount,
                                         IndexSource indexSource,
                                         const void *indices,
                                         uint32_t indexCount,
                                         std::shared_ptr<void> lifetime) {
    BuildGeometry geometry{};
    geometry.vertexSource = VertexSource::ExternalPBR;
    geometry.indexSource = indexSource;
    geometry.externalVertices = vertices;
    geometry.externalIndices = indices;
    geometry.vertexCount = vertexCount;
    geometry.indexCount = indexCount;
    geometry.lifetime = std::move(lifetime);
    return geometry;
}

BuildGeometry BuildGeometry::materializedPBR(std::vector<vk::VertexFormat::EntityPBRVertex> &&vertices,
                                             std::vector<uint32_t> &&indices) {
    BuildGeometry geometry{};
    geometry.vertexSource = VertexSource::MaterializedPBR;
    geometry.indexSource = IndexSource::MaterializedU32;
    geometry.vertexCount = static_cast<uint32_t>(vertices.size());
    geometry.indexCount = static_cast<uint32_t>(indices.size());
    geometry.materializedVertices = std::move(vertices);
    geometry.materializedIndices = std::move(indices);
    return geometry;
}

const vk::VertexFormat::EntityPBRVertex *BuildGeometry::vertexData() const {
    if (inputReleased) { return nullptr; }
    return vertexSource == VertexSource::ExternalPBR ? externalVertices : materializedVertices.data();
}

const vk::VertexFormat::EntityPBRVertex &BuildGeometry::vertex(uint32_t index) const {
    return vertexData()[index];
}

uint32_t BuildGeometry::index(uint32_t index) const {
    switch (indexSource) {
        case IndexSource::GeneratedQuads: return generatedQuadIndex(index);
        case IndexSource::ExternalU16: return static_cast<const uint16_t *>(externalIndices)[index];
        case IndexSource::ExternalU32: return static_cast<const uint32_t *>(externalIndices)[index];
        case IndexSource::MaterializedU32: return materializedIndices[index];
    }
    throw std::runtime_error("Unknown build geometry index source");
}

bool BuildGeometry::empty() const {
    return vertexCount == 0 || indexCount == 0 || vertexData() == nullptr;
}

void BuildGeometry::releaseInput() {
    externalVertices = nullptr;
    externalIndices = nullptr;
    materializedVertices.clear();
    materializedVertices.shrink_to_fit();
    materializedIndices.clear();
    materializedIndices.shrink_to_fit();
    lifetime.reset();
    inputReleased = true;
}

void releaseBuildGeometryInputs(std::vector<BuildGeometry> &geometries) {
    for (auto &geometry : geometries) { geometry.releaseInput(); }
}

void buildPackedGeometry(const std::vector<BuildGeometry> &geometries,
                         std::vector<vk::VertexFormat::EntityPBRVertex> &packedVertices,
                         std::vector<uint32_t> &packedIndices) {
    for (const auto &geometry : geometries) {
        for (uint32_t index = 0; index < geometry.indexCount; ++index) {
            packedIndices.push_back(geometry.index(index));
        }

        const auto *vertices = geometry.vertexData();
        for (uint32_t vertexIndex = 0; vertexIndex < geometry.vertexCount; ++vertexIndex) {
            packedVertices.push_back(vertices[vertexIndex]);
        }
    }
}

uint32_t ChunkBuildGeometry::generatedQuadIndex(uint32_t index) {
    return BuildGeometry::generatedQuadIndex(index);
}

ChunkBuildGeometry ChunkBuildGeometry::externalPBR(const vk::VertexFormat::ChunkPBRVertex *vertices,
                                                   uint32_t vertexCount,
                                                   IndexSource indexSource,
                                                   const void *indices,
                                                   uint32_t indexCount,
                                                   std::shared_ptr<void> lifetime) {
    ChunkBuildGeometry geometry{};
    geometry.vertexSource = VertexSource::ExternalPBR;
    geometry.indexSource = indexSource;
    geometry.externalVertices = vertices;
    geometry.externalIndices = indices;
    geometry.vertexCount = vertexCount;
    geometry.indexCount = indexCount;
    geometry.lifetime = std::move(lifetime);
    return geometry;
}

ChunkBuildGeometry ChunkBuildGeometry::materializedPBR(std::vector<vk::VertexFormat::ChunkPBRVertex> &&vertices,
                                                       std::vector<uint32_t> &&indices) {
    ChunkBuildGeometry geometry{};
    geometry.vertexSource = VertexSource::MaterializedPBR;
    geometry.indexSource = IndexSource::MaterializedU32;
    geometry.vertexCount = static_cast<uint32_t>(vertices.size());
    geometry.indexCount = static_cast<uint32_t>(indices.size());
    geometry.materializedVertices = std::move(vertices);
    geometry.materializedIndices = std::move(indices);
    return geometry;
}

const vk::VertexFormat::ChunkPBRVertex *ChunkBuildGeometry::vertexData() const {
    if (inputReleased) { return nullptr; }
    return vertexSource == VertexSource::ExternalPBR ? externalVertices : materializedVertices.data();
}

const vk::VertexFormat::ChunkPBRVertex &ChunkBuildGeometry::vertex(uint32_t index) const {
    return vertexData()[index];
}

uint32_t ChunkBuildGeometry::index(uint32_t index) const {
    switch (indexSource) {
        case IndexSource::GeneratedQuads: return generatedQuadIndex(index);
        case IndexSource::ExternalU16: return static_cast<const uint16_t *>(externalIndices)[index];
        case IndexSource::ExternalU32: return static_cast<const uint32_t *>(externalIndices)[index];
        case IndexSource::MaterializedU32: return materializedIndices[index];
    }
    throw std::runtime_error("Unknown chunk build geometry index source");
}

bool ChunkBuildGeometry::empty() const {
    return vertexCount == 0 || indexCount == 0 || vertexData() == nullptr;
}

void ChunkBuildGeometry::releaseInput() {
    externalVertices = nullptr;
    externalIndices = nullptr;
    materializedVertices.clear();
    materializedVertices.shrink_to_fit();
    materializedIndices.clear();
    materializedIndices.shrink_to_fit();
    lifetime.reset();
    inputReleased = true;
}

void releaseBuildGeometryInputs(std::vector<ChunkBuildGeometry> &geometries) {
    for (auto &geometry : geometries) { geometry.releaseInput(); }
}

void buildPackedGeometry(const std::vector<ChunkBuildGeometry> &geometries,
                         std::vector<vk::VertexFormat::ChunkPBRVertex> &packedVertices,
                         std::vector<uint32_t> &packedIndices) {
    for (const auto &geometry : geometries) {
        for (uint32_t index = 0; index < geometry.indexCount; ++index) {
            packedIndices.push_back(geometry.index(index));
        }

        const auto *vertices = geometry.vertexData();
        for (uint32_t vertexIndex = 0; vertexIndex < geometry.vertexCount; ++vertexIndex) {
            packedVertices.push_back(vertices[vertexIndex]);
        }
    }
}
