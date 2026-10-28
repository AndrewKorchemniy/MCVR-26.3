#include "common/shared.hpp"

#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTexColor>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionTexColor, position)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionTexColor, uv)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionTexColor, color)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionTexColor>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColor>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionColor, position)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionColor, color)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionColor>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTex>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionTex, position)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionTex, uv)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionTex>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorTexLight>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionColorTexLight, position)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionColorTexLight, color)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionColorTexLight, uv0)},
        {VK_FORMAT_R16G16_SINT, offsetof(VertexFormat::PositionColorTexLight, uv2)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo =
        initVertexLayout<vk::VertexFormat::PositionColorTexLight>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorTexOverlayLightNormal>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, position)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, color)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, uv0)},
        {VK_FORMAT_R16G16_SINT, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, uv1)},
        {VK_FORMAT_R16G16_SINT, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, uv2)},
        {VK_FORMAT_R8G8B8A8_SNORM, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, normal)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo =
        initVertexLayout<vk::VertexFormat::PositionColorTexOverlayLightNormal>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTexColorLight>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionTexColorLight, position)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionTexColorLight, uv0)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionTexColorLight, color)},
        {VK_FORMAT_R16G16_SINT, offsetof(VertexFormat::PositionTexColorLight, uv2)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo =
        initVertexLayout<vk::VertexFormat::PositionTexColorLight>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorNormal>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionColorNormal, position)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionColorNormal, color)},
        {VK_FORMAT_R8G8B8A8_SNORM, offsetof(VertexFormat::PositionColorNormal, normal)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionColorNormal>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorLight>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionColorLight, position)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionColorLight, color)},
        {VK_FORMAT_R16G16_SINT, offsetof(VertexFormat::PositionColorLight, uv2)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionColorLight>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTexLightColor>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionTexLightColor, position)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionTexLightColor, uv0)},
        {VK_FORMAT_R16G16_SINT, offsetof(VertexFormat::PositionTexLightColor, uv2)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionTexLightColor, color)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo =
        initVertexLayout<vk::VertexFormat::PositionTexLightColor>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTexColorNormal>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionTexColorNormal, position)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionTexColorNormal, uv0)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionTexColorNormal, color)},
        {VK_FORMAT_R8G8B8A8_SNORM, offsetof(VertexFormat::PositionTexColorNormal, normal)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo =
        initVertexLayout<vk::VertexFormat::PositionTexColorNormal>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionOnly>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionOnly, position)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionOnly>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::EntityPBRVertex>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::EntityPBRVertex, pos)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::EntityPBRVertex, normalOct)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::EntityPBRVertex, textureUV)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::EntityPBRVertex, colorRGBA8)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::EntityPBRVertex, lightUV)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::EntityPBRVertex, overlayUV)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::EntityPBRVertex, glintUVHalf)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::EntityPBRVertex, textureID)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::EntityPBRVertex, glintTexture)},
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::EntityPBRVertex, postBase)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::EntityPBRVertex, packedData)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::EntityPBRVertex>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::ChunkPBRVertex>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::ChunkPBRVertex, posX)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::ChunkPBRVertex, normalOct)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::ChunkPBRVertex, textureUV)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::ChunkPBRVertex, colorRGBA8)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::ChunkPBRVertex, lightUV)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::ChunkPBRVertex, textureID)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::ChunkPBRVertex, packedData)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::ChunkPBRVertex>(attributes);
    return vertexLayoutInfo;
}
