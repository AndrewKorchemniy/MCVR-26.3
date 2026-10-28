#ifndef VPT_CELESTIAL_BILLBOARD_GLSL
#define VPT_CELESTIAL_BILLBOARD_GLSL

bool intersectCelestialSphere(vec3 rayOrigin, vec3 rayDir, float radius, out float tNear, out float tFar) {
    float b = dot(rayOrigin, rayDir);
    float c = dot(rayOrigin, rayOrigin) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    tNear = -b - h;
    tFar = -b + h;
    return true;
}

void buildCelestialBillboardBasis(in vec3 normal, out vec3 tangent, out vec3 bitangent) {
    float signZ = normal.z >= 0.0 ? 1.0 : -1.0;
    float scale = -1.0 / (signZ + normal.z);
    float crossTerm = normal.x * normal.y * scale;
    tangent = vec3(1.0 + signZ * normal.x * normal.x * scale, signZ * crossTerm, -signZ * normal.x);
    bitangent = vec3(crossTerm, signZ + normal.y * normal.y * scale, -normal.y);
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);
}

vec4 sampleCelestialTextureLod0(sampler2D textureSampler, vec2 textureUv) {
    ivec2 textureSizePixels = textureSize(textureSampler, 0);
    if (textureSizePixels.x <= 0 || textureSizePixels.y <= 0) return vec4(0.0);
    vec2 halfTexel = 0.5 / vec2(textureSizePixels);
    vec2 clampedUv = clamp(textureUv, halfTexel, vec2(1.0) - halfTexel);
    return sampleTexture(textureSampler, clampedUv, 0.0, false);
}

vec4 sampleCelestialAtlasTextureLod0(sampler2D textureSampler, vec2 tileUv, uvec2 tileCount, uvec2 tile) {
    ivec2 textureSizePixels = textureSize(textureSampler, 0);
    if (textureSizePixels.x <= 0 || textureSizePixels.y <= 0) return vec4(0.0);

    vec2 inverseTileCount = 1.0 / vec2(tileCount);
    vec2 tileMin = vec2(tile) * inverseTileCount;
    vec2 tileMax = tileMin + inverseTileCount;
    vec2 halfTexel = 0.5 / vec2(textureSizePixels);
    vec2 minUv = tileMin + halfTexel;
    vec2 maxUv = tileMax - halfTexel;
    vec2 atlasUv = mix(minUv, maxUv, clamp(tileUv, 0.0, 1.0));
    return sampleTexture(textureSampler, atlasUv, 0.0, false);
}

vec4 evalSunBillboard(vec3 rayDir) {
    vec3 sunDir = celestialSunDirection();
    rayDir = normalize(rayDir);
    float z = dot(rayDir, sunDir);
    if (z <= 0.0) return vec4(0.0);

    vec3 right, up;
    buildCelestialBillboardBasis(sunDir, right, up);
    vec2 p = vec2(dot(rayDir, right), dot(rayDir, up));
    vec2 q = p / max(z, 1e-4);
    float tanHalf = tan(0.03);
    vec2 a = abs(q);
    if (a.x > tanHalf || a.y > tanHalf) return vec4(0.0);
    vec2 uv = q / tanHalf * 0.5 + 0.5;
    return sampleCelestialTextureLod0(textures[nonuniformEXT(skyUBO.sunTextureID)], uv);
}

vec4 evalMoonBillboard(vec3 rayDir) {
    vec3 moonDir = celestialMoonDirection();
    rayDir = normalize(rayDir);
    float z = dot(rayDir, moonDir);
    if (z <= 0.0) return vec4(0.0);

    vec3 right, up;
    buildCelestialBillboardBasis(moonDir, right, up);
    vec2 p = vec2(dot(rayDir, right), dot(rayDir, up));
    vec2 q = p / max(z, 1e-4);
    float tanHalf = tan(0.05);
    vec2 a = abs(q);
    if (a.x > tanHalf || a.y > tanHalf) return vec4(0.0);
    vec2 uv = q / tanHalf * 0.5 + 0.5;
    // 26.3 has one texture per moon phase (the current phase's id is in moonTextureID), so the whole texture is the moon.
    uvec2 tileCount = uvec2(1u, 1u);
    uvec2 tile = uvec2(0u, 0u);
    return sampleCelestialAtlasTextureLod0(textures[nonuniformEXT(skyUBO.moonTextureID)], uv, tileCount, tile);
}

bool isCelestialBillboardVisible(vec3 rayDirection) {
    if (worldUBO.skyType != 1) return false;
    if (skyUBO.cameraSubmersionType == 0 || skyUBO.cameraSubmersionType == 2 || skyUBO.hasBlindnessOrDarkness > 0) {
        return false;
    }

    float rayLengthSquared = dot(rayDirection, rayDirection);
    if (rayLengthSquared <= 1e-8 || isnan(rayLengthSquared) || isinf(rayLengthSquared)) return false;
    vec3 rayDir = rayDirection * inversesqrt(rayLengthSquared);
    float cameraHeight = worldUBO.cameraViewMatInv[3].y;
    vec3 planetPosition = vec3(0.0, VPT_ATMOSPHERE_RG + cameraHeight + 70.0, 0.0);
    float groundNear;
    float groundFar;
    bool blockedByGround =
        intersectCelestialSphere(planetPosition, rayDir, VPT_ATMOSPHERE_RG, groundNear, groundFar) && groundFar > 1e-3;
    if (blockedByGround) return false;

    return evalSunBillboard(rayDir).a > 1e-4 || evalMoonBillboard(rayDir).a > 1e-4;
}

#endif
