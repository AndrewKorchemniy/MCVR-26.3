#include "core/vulkan/vertex.hpp"

#include "common/shared.hpp"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>

bool vk::Vertex::hasFlag(uint32_t packedData, uint32_t flag) {
    return (packedData & flag) != 0u;
}

uint32_t vk::Vertex::setFlag(uint32_t packedData, uint32_t flag) {
    return packedData | flag;
}

uint32_t vk::Vertex::setCoordinate(uint32_t packedData, uint32_t coordinate) {
    return (packedData & ~(0xFu << coordinateShift)) | ((coordinate & 0xFu) << coordinateShift);
}

uint32_t vk::Vertex::setAlphaMode(uint32_t packedData, uint32_t alphaMode) {
    return (packedData & ~(0xFu << alphaModeShift)) | ((alphaMode & 0xFu) << alphaModeShift);
}

uint32_t vk::Vertex::setAlbedoEmission(uint32_t packedData, float emission) {
    uint32_t packedHalf = glm::packHalf2x16(glm::vec2(std::max(emission, 0.0f), 0.0f)) & 0xffffu;
    return (packedData & 0x0000ffffu) | (packedHalf << albedoEmissionShift);
}

float vk::Vertex::unpackAlbedoEmission(uint32_t packedData) {
    return glm::unpackHalf2x16(packedData >> albedoEmissionShift).x;
}

glm::vec2 vk::Vertex::octSignNotZero(const glm::vec2 &value) {
    return {
        value.x >= 0.0f ? 1.0f : -1.0f,
        value.y >= 0.0f ? 1.0f : -1.0f,
    };
}

uint32_t vk::Vertex::packU16x2(int x, int y) {
    return (static_cast<uint32_t>(x) & 0xffffu) | ((static_cast<uint32_t>(y) & 0xffffu) << 16u);
}

uint32_t vk::Vertex::packU16x2(uint32_t packed) {
    return packed;
}

glm::ivec2 vk::Vertex::unpackU16x2(uint32_t packed) {
    return {
        static_cast<int32_t>(packed & 0xffffu),
        static_cast<int32_t>((packed >> 16u) & 0xffffu),
    };
}

uint32_t vk::Vertex::packColorRGBA8(uint32_t rgba8) {
    return rgba8;
}

uint32_t vk::Vertex::packColorRGBA8(const glm::vec4 &color) {
    return glm::packUnorm4x8(glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0f)));
}

glm::vec4 vk::Vertex::unpackColorRGBA8(uint32_t rgba8) {
    return glm::unpackUnorm4x8(rgba8);
}

uint32_t vk::Vertex::packNormalOct(const glm::vec3 &normal) {
    const float length = glm::length(normal);
    if (length <= 1e-8f || !std::isfinite(length)) { return 0u; }

    glm::vec3 unit = normal / length;
    unit /= std::max(std::abs(unit.x) + std::abs(unit.y) + std::abs(unit.z), 1e-8f);
    glm::vec2 encoded = glm::vec2(unit.x, unit.y);
    if (unit.z < 0.0f) {
        encoded = (glm::vec2(1.0f) - glm::abs(glm::vec2(encoded.y, encoded.x))) * octSignNotZero(encoded);
    }
    return glm::packSnorm2x16(glm::clamp(encoded, glm::vec2(-1.0f), glm::vec2(1.0f)));
}

uint32_t vk::Vertex::packNormalBytes(uint32_t packedNormalBytes) {
    glm::vec3 normal = {
        static_cast<float>(static_cast<int8_t>(packedNormalBytes & 0xffu)),
        static_cast<float>(static_cast<int8_t>((packedNormalBytes >> 8u) & 0xffu)),
        static_cast<float>(static_cast<int8_t>((packedNormalBytes >> 16u) & 0xffu)),
    };
    return packNormalOct(normal);
}

glm::vec3 vk::Vertex::unpackNormalOct(uint32_t packedNormal) {
    glm::vec2 encoded = glm::unpackSnorm2x16(packedNormal);
    glm::vec3 normal = glm::vec3(encoded, 1.0f - std::abs(encoded.x) - std::abs(encoded.y));
    if (normal.z < 0.0f) {
        normal.x = (1.0f - std::abs(encoded.y)) * (encoded.x >= 0.0f ? 1.0f : -1.0f);
        normal.y = (1.0f - std::abs(encoded.x)) * (encoded.y >= 0.0f ? 1.0f : -1.0f);
    }
    const float length = glm::length(normal);
    return length > 1e-8f && std::isfinite(length) ? normal / length : glm::vec3(0.0f, 0.0f, 1.0f);
}

uint32_t vk::Vertex::packHalf2(const glm::vec2 &value) {
    return glm::packHalf2x16(value);
}

glm::vec2 vk::Vertex::unpackHalf2(uint32_t packed) {
    return glm::unpackHalf2x16(packed);
}

void vk::Vertex::setColorLayer(VertexFormat::EntityPBRVertex &vertex, uint32_t rgba8) {
    vertex.packedData = setFlag(vertex.packedData, useColorLayerBit);
    vertex.colorRGBA8 = packColorRGBA8(rgba8);
}

void vk::Vertex::setColorLayer(VertexFormat::EntityPBRVertex &vertex, const glm::vec4 &color) {
    vertex.packedData = setFlag(vertex.packedData, useColorLayerBit);
    vertex.colorRGBA8 = packColorRGBA8(color);
}

void vk::Vertex::setTexture(VertexFormat::EntityPBRVertex &vertex, const glm::vec2 &uv) {
    vertex.packedData = setFlag(vertex.packedData, useTextureBit);
    vertex.textureUV = uv;
}

void vk::Vertex::setOverlay(VertexFormat::EntityPBRVertex &vertex, uint32_t packedUv) {
    vertex.packedData = setFlag(vertex.packedData, useOverlayBit);
    vertex.overlayUV = packU16x2(packedUv);
}

void vk::Vertex::setLight(VertexFormat::EntityPBRVertex &vertex, uint32_t packedUv) {
    vertex.packedData = setFlag(vertex.packedData, useLightBit);
    vertex.lightUV = packU16x2(packedUv);
}

void vk::Vertex::setNormal(VertexFormat::EntityPBRVertex &vertex, const glm::vec3 &normal) {
    vertex.packedData = setFlag(vertex.packedData, useNormBit);
    vertex.normalOct = packNormalOct(normal);
}

void vk::Vertex::setNormalBytes(VertexFormat::EntityPBRVertex &vertex, uint32_t packedNormalBytes) {
    vertex.packedData = setFlag(vertex.packedData, useNormBit);
    vertex.normalOct = packNormalBytes(packedNormalBytes);
}

void vk::Vertex::setGlint(VertexFormat::EntityPBRVertex &vertex, uint32_t glintTexture, const glm::vec2 &uv) {
    vertex.packedData = setFlag(vertex.packedData, useGlintBit);
    vertex.glintTexture = glintTexture;
    vertex.glintUVHalf = packHalf2(uv);
}

void vk::Vertex::applyNormalOffset(VertexFormat::EntityPBRVertex &vertex) {
    applyNormalOffset(vertex, 0.00001f);
}

void vk::Vertex::applyNormalOffset(VertexFormat::EntityPBRVertex &vertex, float distance) {
    if (!hasFlag(vertex.packedData, useNormBit)) { return; }
    vertex.pos += distance * unpackNormalOct(vertex.normalOct);
}

void vk::Vertex::setCoordinate(VertexFormat::EntityPBRVertex &vertex, uint32_t coordinate) {
    vertex.packedData = setCoordinate(vertex.packedData, coordinate);
}

void vk::Vertex::setPostBase(VertexFormat::EntityPBRVertex &vertex, double x, double y, double z) {
    vertex.postBase = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
}

uint32_t vk::Vertex::packMaterialFlags(const VertexFormat::EntityPBRVertex &vertex) {
    return vertex.packedData;
}

vk::VertexFormat::PositionVertex vk::Vertex::makePositionVertex(const VertexFormat::EntityPBRVertex &vertex) {
    return {
        .pos = vertex.pos,
        .pad0 = 0,
    };
}

vk::VertexFormat::MaterialVertex vk::Vertex::makeMaterialVertex(const VertexFormat::EntityPBRVertex &vertex) {
    return {
        .norm = unpackNormalOct(vertex.normalOct),
        .textureID = vertex.textureID,
        .colorLayer = unpackColorRGBA8(vertex.colorRGBA8),
        .textureUV = vertex.textureUV,
        .overlayUV = unpackU16x2(vertex.overlayUV),
        .glintUV = unpackHalf2(vertex.glintUVHalf),
        .glintTexture = vertex.glintTexture,
        .albedoEmission = unpackAlbedoEmission(vertex.packedData),
        .lightUV = unpackU16x2(vertex.lightUV),
        .packedData = packMaterialFlags(vertex),
        .pad0 = 0,
    };
}

std::vector<vk::VertexFormat::PositionVertex>
vk::Vertex::buildPositionVertices(const std::vector<VertexFormat::EntityPBRVertex> &vertices) {
    std::vector<VertexFormat::PositionVertex> packedVertices;
    packedVertices.reserve(vertices.size());
    for (const auto &vertex : vertices) { packedVertices.push_back(makePositionVertex(vertex)); }
    return packedVertices;
}

std::vector<vk::VertexFormat::MaterialVertex>
vk::Vertex::buildMaterialVertices(const std::vector<VertexFormat::EntityPBRVertex> &vertices) {
    std::vector<VertexFormat::MaterialVertex> packedVertices;
    packedVertices.reserve(vertices.size());
    for (const auto &vertex : vertices) { packedVertices.push_back(makeMaterialVertex(vertex)); }
    return packedVertices;
}

glm::vec3 vk::Vertex::position(const VertexFormat::ChunkPBRVertex &vertex) {
    return {vertex.posX, vertex.posY, vertex.posZ};
}

void vk::Vertex::setPosition(VertexFormat::ChunkPBRVertex &vertex, const glm::vec3 &position) {
    vertex.posX = position.x;
    vertex.posY = position.y;
    vertex.posZ = position.z;
}

uint32_t vk::Vertex::packMaterialFlags(const VertexFormat::ChunkPBRVertex &vertex) {
    return vertex.packedData & ~(useOverlayBit | useGlintBit);
}

vk::VertexFormat::PositionVertex vk::Vertex::makePositionVertex(const VertexFormat::ChunkPBRVertex &vertex) {
    return {
        .pos = position(vertex),
        .pad0 = 0,
    };
}

vk::VertexFormat::MaterialVertex vk::Vertex::makeMaterialVertex(const VertexFormat::ChunkPBRVertex &vertex) {
    const uint32_t packedData = packMaterialFlags(vertex);
    return {
        .norm = unpackNormalOct(vertex.normalOct),
        .textureID = vertex.textureID,
        .colorLayer = unpackColorRGBA8(vertex.colorRGBA8),
        .textureUV = vertex.textureUV,
        .overlayUV = glm::ivec2(0),
        .glintUV = glm::vec2(0.0f),
        .glintTexture = 0,
        .albedoEmission = unpackAlbedoEmission(packedData),
        .lightUV = unpackU16x2(vertex.lightUV),
        .packedData = packedData,
        .pad0 = 0,
    };
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::Triangle>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(vk::VertexFormat::Triangle, pos)},
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(vk::VertexFormat::Triangle, color)},
    };

    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::Triangle>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::TexturedTriangle>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(vk::VertexFormat::TexturedTriangle, pos)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(vk::VertexFormat::TexturedTriangle, uv)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::TexturedTriangle>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::ArrayTexturedTriangle>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(vk::VertexFormat::ArrayTexturedTriangle, pos)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(vk::VertexFormat::ArrayTexturedTriangle, uv)},
        {VK_FORMAT_R32_SFLOAT, offsetof(vk::VertexFormat::ArrayTexturedTriangle, textureLayer)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo =
        initVertexLayout<vk::VertexFormat::ArrayTexturedTriangle>(attributes);
    return vertexLayoutInfo;
}
