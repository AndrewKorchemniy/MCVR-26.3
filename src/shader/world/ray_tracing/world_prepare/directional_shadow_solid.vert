#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common/shared.hpp"

struct AccelerationStructureInstance {
    vec4 transform0;
    vec4 transform1;
    vec4 transform2;
    uint customIndexAndMask;
    uint sbtOffsetAndFlags;
    uint referenceLow;
    uint referenceHigh;
};

layout(push_constant) uniform DirectionalShadowPushConstants {
    mat4 lightViewProj;
}
shadowPush;

layout(set = 1, binding = 10, std430) readonly buffer TlasInstances {
    AccelerationStructureInstance instances[];
}
tlasInstances;

layout(set = 1, binding = 2) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(std430, buffer_reference, buffer_reference_align = 4) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "../../../util/vertex.glsl"

struct RenderIndirectMetadataHeader {
    uint drawCount;
    uint metadataOffsetBytes;
    uint metadataStrideBytes;
    uint commandStrideBytes;
    uint chunkInstanceIndexBase;
    uint chunkGeometryAddressIndexBase;
    uint entityInstanceCount;
    uint entityGeometryCount;
    uint chunkInstanceCount;
    uint chunkGeometrySlotCount;
};

struct RenderIndirectDrawMetadata {
    uint objectType;
    uint objectIndex;
    uint geometryIndex;
    uint instanceIndex;
    uint geometryAddressIndex;
    uint vertexCount;
    uint indexCount;
    uint layerOrder;
    uint renderLayerHashLow;
    uint renderLayerHashHigh;
    uint flags;
    uint reserved;
};

layout(set = 1, binding = 26, std430) readonly buffer DirectionalShadowIndirectMetadata {
    RenderIndirectMetadataHeader header;
    RenderIndirectDrawMetadata metadata[];
}
directionalShadowMetadata;

const float DIRECTIONAL_SHADOW_RADIAL_DISTORTION = 0.0;

vec3 transformPoint(AccelerationStructureInstance instance, vec3 position) {
    return vec3(dot(instance.transform0.xyz, position) + instance.transform0.w,
                dot(instance.transform1.xyz, position) + instance.transform1.w,
                dot(instance.transform2.xyz, position) + instance.transform2.w);
}

vec4 applyDirectionalShadowWarp(vec4 clipPosition) {
    if (abs(clipPosition.w) <= 1e-8) { return clipPosition; }

    vec2 ndc = clipPosition.xy / clipPosition.w;
    float radial = clamp(length(ndc), 0.0, 1.0);
    float bias = 1.0 - DIRECTIONAL_SHADOW_RADIAL_DISTORTION + radial * DIRECTIONAL_SHADOW_RADIAL_DISTORTION;
    clipPosition.xy = ndc / max(bias, 0.25) * clipPosition.w;
    return clipPosition;
}

bool resolveDirectionalShadowDrawMetadata(RenderIndirectDrawMetadata drawMetadata,
                                          out uint instanceIndex,
                                          out uint geometryBufferIndex) {
    instanceIndex = drawMetadata.instanceIndex;
    geometryBufferIndex = drawMetadata.geometryAddressIndex;
    if (drawMetadata.objectType == 0u) {
        if (instanceIndex >= directionalShadowMetadata.header.chunkInstanceCount ||
            geometryBufferIndex >= directionalShadowMetadata.header.chunkGeometrySlotCount) {
            return false;
        }
        instanceIndex += directionalShadowMetadata.header.chunkInstanceIndexBase;
        geometryBufferIndex = markChunkGeometryBufferIndex(
            geometryBufferIndex + directionalShadowMetadata.header.chunkGeometryAddressIndexBase);
        return true;
    }
    return instanceIndex < directionalShadowMetadata.header.entityInstanceCount &&
           geometryBufferIndex < directionalShadowMetadata.header.entityGeometryCount;
}

void main() {
    uint drawIndex = gl_BaseInstance;
    RenderIndirectDrawMetadata drawMetadata = directionalShadowMetadata.metadata[drawIndex];
    uint instanceIndex;
    uint geometryBufferIndex;
    if (!resolveDirectionalShadowDrawMetadata(drawMetadata, instanceIndex, geometryBufferIndex)) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    if (drawMetadata.vertexCount == 0u || gl_VertexIndex >= drawMetadata.indexCount) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    IndexBuffer indexBuffer = IndexBuffer(indexBufferAddrs.addrs[geometryAddressIndex(geometryBufferIndex)]);
    uint localVertexIndex = indexBuffer.indices[gl_VertexIndex];
    if (localVertexIndex >= drawMetadata.vertexCount) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    AccelerationStructureInstance instance = tlasInstances.instances[instanceIndex];
    vec3 relativePosition = transformPoint(instance, loadVertexPosition(geometryBufferIndex, localVertexIndex));
    gl_Position = applyDirectionalShadowWarp(shadowPush.lightViewProj * vec4(relativePosition, 1.0));
}
