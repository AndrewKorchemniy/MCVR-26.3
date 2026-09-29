#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "alpha_mode.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(location = 0) flat in uint textureId;
layout(location = 1) in vec4 colorLayer;
layout(location = 2) in vec2 textureUv;
layout(location = 3) flat in uint packedData;
layout(location = 4) flat in uvec2 renderLayerHash;

layout(location = 0) out vec4 outTint;

const uint USE_COLOR_LAYER_BIT = 1u << 0u;
const uint USE_TEXTURE_BIT = 1u << 1u;

const uvec2 RENDER_LAYER_HASH_TRANSLUCENT = uvec2(1847089516u, 3557394643u);
const uvec2 RENDER_LAYER_HASH_TRANSLUCENT_MOVING_BLOCK = uvec2(2892334413u, 3447226570u);
const uvec2 RENDER_LAYER_HASH_TRIPWIRE = uvec2(3613336919u, 2969457857u);
const uvec2 RENDER_LAYER_HASH_ITEM_ENTITY_TRANSLUCENT_CULL = uvec2(2097292005u, 128709809u);
const uvec2 RENDER_LAYER_HASH_ENTITY_TRANSLUCENT = uvec2(1506745222u, 2510760922u);
const uvec2 RENDER_LAYER_HASH_ENTITY_TRANSLUCENT_EMISSIVE = uvec2(3402099520u, 972664649u);
const uvec2 RENDER_LAYER_HASH_ARMOR_TRANSLUCENT = uvec2(496475400u, 4123064749u);
const uvec2 RENDER_LAYER_HASH_ENTITY_ALPHA = uvec2(2461434209u, 101306947u);
const uvec2 RENDER_LAYER_HASH_EYES = uvec2(2462904747u, 4019224416u);

bool hasColorLayer() {
    return (packedData & USE_COLOR_LAYER_BIT) != 0u;
}

bool hasTexture() {
    return (packedData & USE_TEXTURE_BIT) != 0u;
}

uint alphaMode() {
    return getSurfaceAlphaMode(packedData);
}

bool hashEquals(uvec2 value, uvec2 expected) {
    return value.x == expected.x && value.y == expected.y;
}

bool isTransparentRenderLayer() {
    return hashEquals(renderLayerHash, RENDER_LAYER_HASH_TRANSLUCENT) ||
           hashEquals(renderLayerHash, RENDER_LAYER_HASH_TRANSLUCENT_MOVING_BLOCK) ||
           hashEquals(renderLayerHash, RENDER_LAYER_HASH_TRIPWIRE) ||
           hashEquals(renderLayerHash, RENDER_LAYER_HASH_ITEM_ENTITY_TRANSLUCENT_CULL) ||
           hashEquals(renderLayerHash, RENDER_LAYER_HASH_ENTITY_TRANSLUCENT) ||
           hashEquals(renderLayerHash, RENDER_LAYER_HASH_ENTITY_TRANSLUCENT_EMISSIVE) ||
           hashEquals(renderLayerHash, RENDER_LAYER_HASH_ARMOR_TRANSLUCENT) ||
           hashEquals(renderLayerHash, RENDER_LAYER_HASH_ENTITY_ALPHA) ||
           hashEquals(renderLayerHash, RENDER_LAYER_HASH_EYES);
}

void main() {
    vec4 layerColor = hasColorLayer() ? colorLayer : vec4(1.0);
    vec4 albedo = hasTexture() ? resolveSurfaceTextureColor(texture(textures[nonuniformEXT(textureId)], textureUv), alphaMode()) : vec4(1.0);
    float alpha = resolveSurfaceAlpha(albedo.a * layerColor.a, alphaMode());
    if (alpha <= 0.01) { discard; }

    bool transparentLayer = isTransparentRenderLayer();
    bool transparent = transparentLayer || alphaMode() == ALPHA_MODE_TRANSPARENT || alpha < 0.98;
    if (!transparent) { discard; }

    vec3 tint = clamp(albedo.rgb * layerColor.rgb, vec3(0.0), vec3(1.0));
    float layerLimitedAlpha = transparentLayer && alpha > 0.75 ? 0.45 : alpha;
    float shadowAlpha = clamp(layerLimitedAlpha * 0.72, 0.02, 0.85);
    vec3 shadowTint = mix(vec3(1.0), tint, 0.85);
    outTint = vec4(shadowTint, shadowAlpha);
}
