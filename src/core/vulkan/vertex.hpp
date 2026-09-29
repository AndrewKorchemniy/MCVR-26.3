#pragma once

#include "common/shared.hpp"
#include "core/all_extern.hpp"

#include <vector>

namespace vk {
struct VertexLayoutInfo {
    VkVertexInputBindingDescription bindingDescription;
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
};

struct VertexAttribute {
    VkFormat format;
    uint32_t offset;
};

struct Vertex {
    static constexpr uint32_t useColorLayerBit = 1u << 0u;
    static constexpr uint32_t useTextureBit = 1u << 1u;
    static constexpr uint32_t useOverlayBit = 1u << 2u;
    static constexpr uint32_t useGlintBit = 1u << 3u;
    static constexpr uint32_t useNormBit = 1u << 4u;
    static constexpr uint32_t useLightBit = 1u << 5u;
    static constexpr uint32_t noHeightSurfaceBit = 1u << 6u;
    static constexpr uint32_t alphaModeShift = 8u;
    static constexpr uint32_t coordinateShift = 12u;
    static constexpr uint32_t albedoEmissionShift = 16u;

    template <typename T>
    static VertexLayoutInfo initVertexLayout(std::vector<VertexAttribute> &attributes);

    template <typename T>
    static void buildVertexLayoutInfo(VertexLayoutInfo &vertexLayoutInfo, std::vector<VertexAttribute> &attributes);

    template <typename T>
    static VertexLayoutInfo &vertexLayoutInfo();

    static bool hasFlag(uint32_t packedData, uint32_t flag);
    static uint32_t setFlag(uint32_t packedData, uint32_t flag);
    static uint32_t setCoordinate(uint32_t packedData, uint32_t coordinate);
    static uint32_t setAlphaMode(uint32_t packedData, uint32_t alphaMode);
    static uint32_t setAlbedoEmission(uint32_t packedData, float emission);
    static float unpackAlbedoEmission(uint32_t packedData);
    static glm::vec2 octSignNotZero(const glm::vec2 &value);

    static uint32_t packU16x2(int x, int y);
    static uint32_t packU16x2(uint32_t packed);
    static glm::ivec2 unpackU16x2(uint32_t packed);

    static uint32_t packColorRGBA8(uint32_t rgba8);
    static uint32_t packColorRGBA8(const glm::vec4 &color);
    static glm::vec4 unpackColorRGBA8(uint32_t rgba8);

    static uint32_t packNormalOct(const glm::vec3 &normal);
    static uint32_t packNormalBytes(uint32_t packedNormalBytes);
    static glm::vec3 unpackNormalOct(uint32_t packedNormal);

    static uint32_t packHalf2(const glm::vec2 &value);
    static glm::vec2 unpackHalf2(uint32_t packed);

    static void setColorLayer(VertexFormat::EntityPBRVertex &vertex, uint32_t rgba8);
    static void setColorLayer(VertexFormat::EntityPBRVertex &vertex, const glm::vec4 &color);
    static void setTexture(VertexFormat::EntityPBRVertex &vertex, const glm::vec2 &uv);
    static void setOverlay(VertexFormat::EntityPBRVertex &vertex, uint32_t packedUv);
    static void setLight(VertexFormat::EntityPBRVertex &vertex, uint32_t packedUv);
    static void setNormal(VertexFormat::EntityPBRVertex &vertex, const glm::vec3 &normal);
    static void setNormalBytes(VertexFormat::EntityPBRVertex &vertex, uint32_t packedNormalBytes);
    static void setGlint(VertexFormat::EntityPBRVertex &vertex, uint32_t glintTexture, const glm::vec2 &uv);
    static void applyNormalOffset(VertexFormat::EntityPBRVertex &vertex);
    static void applyNormalOffset(VertexFormat::EntityPBRVertex &vertex, float distance);
    static void setCoordinate(VertexFormat::EntityPBRVertex &vertex, uint32_t coordinate);
    static void setPostBase(VertexFormat::EntityPBRVertex &vertex, double x, double y, double z);

    static uint32_t packMaterialFlags(const VertexFormat::EntityPBRVertex &vertex);
    static VertexFormat::PositionVertex makePositionVertex(const VertexFormat::EntityPBRVertex &vertex);
    static VertexFormat::MaterialVertex makeMaterialVertex(const VertexFormat::EntityPBRVertex &vertex);
    static std::vector<VertexFormat::PositionVertex>
    buildPositionVertices(const std::vector<VertexFormat::EntityPBRVertex> &vertices);
    static std::vector<VertexFormat::MaterialVertex>
    buildMaterialVertices(const std::vector<VertexFormat::EntityPBRVertex> &vertices);

    static glm::vec3 position(const VertexFormat::ChunkPBRVertex &vertex);
    static void setPosition(VertexFormat::ChunkPBRVertex &vertex, const glm::vec3 &position);
    static uint32_t packMaterialFlags(const VertexFormat::ChunkPBRVertex &vertex);
    static VertexFormat::PositionVertex makePositionVertex(const VertexFormat::ChunkPBRVertex &vertex);
    static VertexFormat::MaterialVertex makeMaterialVertex(const VertexFormat::ChunkPBRVertex &vertex);
};

template <typename T>
vk::VertexLayoutInfo vk::Vertex::initVertexLayout(std::vector<VertexAttribute> &attributes) {
    vk::VertexLayoutInfo vertexLayoutInfo{};
    buildVertexLayoutInfo<T>(vertexLayoutInfo, attributes);
    return vertexLayoutInfo;
}

template <typename T>
void Vertex::buildVertexLayoutInfo(VertexLayoutInfo &vertexLayoutInfo, std::vector<VertexAttribute> &attributes) {
    vertexLayoutInfo.bindingDescription.binding = 0;
    vertexLayoutInfo.bindingDescription.stride = sizeof(T);
    vertexLayoutInfo.bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    for (uint32_t i = 0; i < attributes.size(); i++) {
        vertexLayoutInfo.attributeDescriptions.push_back({
            .location = i,
            .binding = 0,
            .format = attributes[i].format,
            .offset = attributes[i].offset,
        });
    }
}
}
