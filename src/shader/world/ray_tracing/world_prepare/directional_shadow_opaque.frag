#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "alpha_mode.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(location = 0) flat in uint textureId;
layout(location = 1) in vec4 colorLayer;
layout(location = 2) in vec2 textureUv;
layout(location = 3) flat in uint packedData;

const uint USE_COLOR_LAYER_BIT = 1u << 0u;
const uint USE_TEXTURE_BIT = 1u << 1u;

bool hasColorLayer() {
    return (packedData & USE_COLOR_LAYER_BIT) != 0u;
}

bool hasTexture() {
    return (packedData & USE_TEXTURE_BIT) != 0u;
}

void main() {
    uint alphaMode = getSurfaceAlphaMode(packedData);
    vec4 layerColor = hasColorLayer() ? colorLayer : vec4(1.0);
    vec4 albedo = hasTexture() ? resolveSurfaceTextureColor(texture(textures[nonuniformEXT(textureId)], textureUv), alphaMode) : vec4(1.0);
    float alpha = resolveSurfaceAlpha(albedo.a * layerColor.a, alphaMode);
    if (alpha <= 0.01) { discard; }
}
