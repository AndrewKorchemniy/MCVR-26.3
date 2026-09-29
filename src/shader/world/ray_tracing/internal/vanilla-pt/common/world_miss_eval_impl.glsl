#include "common/shared.hpp"
#include "util/ray.glsl"
#include "util/util.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(set = 5, binding = 0) uniform sampler2D transLUT;
layout(set = 5, binding = 2) uniform samplerCube skyFull;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(location = 0) rayPayloadInEXT MainRay mainRay;

#include "common/volumetric_cloud.glsl"
#include "common/celestial.glsl"
#include "common/celestial_billboard.glsl"

void main() {
    mainRay.directLightRadiance.x = 1.0;

    if (skyUBO.cameraSubmersionType == 0 || skyUBO.cameraSubmersionType == 2 || skyUBO.hasBlindnessOrDarkness > 0) {
        raySetStop(mainRay, true);
        mainRay.hitT = INF_DISTANCE;
        return;
    }

    switch (worldUBO.skyType) {
        case 0:
        case 2:
            raySetStop(mainRay, true);
            mainRay.hitT = INF_DISTANCE;
            return;
        case 1:
        default: break;
    }

    vec3 rayDir = normalize(gl_WorldRayDirectionEXT);
    vec3 sunDir = celestialSunDirection();
    float progress = skyUBO.rainGradient;
    vec3 rainyRadiance = mix(vec3(0.0), vec3(0.1), smoothstep(-0.3, 0.3, sunDir.y));
    vec3 backgroundRadiance = mix(texture(skyFull, rayDir).rgb, rainyRadiance, progress);

    if (worldUBO.skyType == 1) {
        float cameraHeight = worldUBO.cameraViewMatInv[3].y;
        vec3 pPlanet = vec3(0.0, VPT_ATMOSPHERE_RG + cameraHeight + 70.0, 0.0);
        float r = clamp(length(pPlanet), VPT_ATMOSPHERE_RG, VPT_ATMOSPHERE_RT);
        vec3 up = pPlanet / max(r, 1e-6);
        float mu = clamp(dot(up, rayDir), -1.0, 1.0);
        vec3 transmittance = sampleCloudAtmosphereTransmittance(r, mu);

        vec4 sunSample = evalSunBillboard(rayDir);
        if (sunSample.a > 1e-4) {
            float tG0, tG1;
            bool hitGround = intersectCelestialSphere(pPlanet, rayDir, VPT_ATMOSPHERE_RG, tG0, tG1);
            bool blocked = hitGround && (tG1 > 1e-3);
            if (!blocked) {
                vec3 sunRadiance = sunSample.rgb * VPT_SUN_RADIANCE * transmittance * sunSample.a;
                backgroundRadiance += mix(sunRadiance, vec3(0.0), progress);
            }
        }

        vec4 moonSample = evalMoonBillboard(rayDir);
        if (moonSample.a > 1e-4) {
            float tG0, tG1;
            bool hitGround = intersectCelestialSphere(pPlanet, rayDir, VPT_ATMOSPHERE_RG, tG0, tG1);
            bool blocked = hitGround && (tG1 > 1e-3);
            if (!blocked) {
                vec3 moonRadiance = moonSample.rgb * VPT_MOON_RADIANCE * max(transmittance, vec3(0.03));
                backgroundRadiance += mix(moonRadiance, vec3(0.0), progress);
            }
        }
    }

#if VPT_ALLOW_VOLUMETRIC_CLOUD_MISS
    if (VPT_CLOUD_MODE == 2u) {
        VolumetricCloudResult cloudResult =
            rayUseIndirectVolumetricCloud(mainRay) ?
                applyVolumetricCloudBudgeted(
                    gl_WorldRayOriginEXT, rayDir, backgroundRadiance, VPT_INDIRECT_VOLUMETRIC_CLOUD_VIEW_STEPS,
                    VPT_INDIRECT_VOLUMETRIC_CLOUD_LIGHT_STEPS, VPT_INDIRECT_VOLUMETRIC_CLOUD_AMBIENT_STEPS) :
                applyVolumetricCloud(gl_WorldRayOriginEXT, rayDir, backgroundRadiance);
        backgroundRadiance = cloudResult.color;
        vec3 clampedBackground = max(backgroundRadiance, vec3(0.0));
        vec3 cloudOnly = max(cloudResult.color - clampedBackground * cloudResult.transmittance, vec3(0.0));
        float backgroundLum = volumetricCloudLuminance(clampedBackground);
        float cloudLum = volumetricCloudLuminance(cloudOnly);
        float cloudPresence = cloudLum / max(backgroundLum + cloudLum, 1e-4);
        float starVisibility = min(cloudResult.transmittance, 1.0 - cloudPresence * 1.35);
        mainRay.directLightRadiance.x = cloudSaturate(starVisibility);
    }
#endif

    mainRay.radiance += backgroundRadiance * mainRay.throughput;
    raySetStop(mainRay, true);
    mainRay.hitT = INF_DISTANCE;
}
