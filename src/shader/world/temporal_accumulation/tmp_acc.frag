#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D texCurrent;
layout(set = 0, binding = 1) uniform sampler2D texHistory;
layout(set = 0, binding = 2) uniform sampler2D texMotion;

layout(set = 0, binding = 3) uniform sampler2D texNormalCurrent;
layout(set = 0, binding = 4) uniform sampler2D texNormalHistory;
layout(set = 0, binding = 5) uniform sampler2D texDepthCurrent;
layout(set = 0, binding = 6) uniform sampler2D texDepthHistory;

layout(push_constant) uniform PushConstants {
    float alpha;
    float normalThreshold;
    float depthThreshold;
    float depthScale;
}
pushConstants;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragNormal;
layout(location = 2) out float fragDepth;

bool validDepth(float depth) {
    return !isnan(depth) && !isinf(depth) && depth > 0.0 && depth < 65000.0;
}

vec3 safeNormalize(vec3 value, vec3 fallback) {
    float len2 = dot(value, value);
    return len2 > 1e-8 ? value * inversesqrt(len2) : fallback;
}

void main() {
    ivec2 size = textureSize(texCurrent, 0);
    vec2 resolution = vec2(size);
    vec2 texelSize = 1.0 / resolution;

    vec2 velocityPixels = texture(texMotion, texCoord).xy;
    vec2 velocityUV = velocityPixels / resolution;
    vec2 historyUV = texCoord + velocityUV;

    vec3 colorCurrent = texture(texCurrent, texCoord).rgb;
    vec3 normalCurrent = texture(texNormalCurrent, texCoord).xyz;
    float depthCurrent = texture(texDepthCurrent, texCoord).r;

    vec3 minColor = vec3(10000.0);
    vec3 maxColor = vec3(-10000.0);

    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 neighborUV = texCoord + vec2(x, y) * texelSize;
            vec3 neighborColor = texture(texCurrent, neighborUV).rgb;

            minColor = min(minColor, neighborColor);
            maxColor = max(maxColor, neighborColor);
        }
    }

    bool isOffScreen = any(lessThan(historyUV, vec2(0.0))) || any(greaterThan(historyUV, vec2(1.0)));

    vec3 colorHistory = colorCurrent;
    vec3 normalHistory = normalCurrent;
    float depthHistory = depthCurrent;

    bool rejectHistory = isOffScreen;

    if (!isOffScreen) {
        normalHistory = texture(texNormalHistory, historyUV).xyz;
        depthHistory = texture(texDepthHistory, historyUV).r;

        bool useNormalReject = dot(normalCurrent, normalCurrent) > 1e-6 && dot(normalHistory, normalHistory) > 1e-6;
        float normalSimilarity =
            dot(safeNormalize(normalCurrent, vec3(0.0, 1.0, 0.0)), safeNormalize(normalHistory, vec3(0.0, 1.0, 0.0)));
        bool rejectNormal = useNormalReject && normalSimilarity < pushConstants.normalThreshold;

        bool rejectDepth = false;
        if (validDepth(depthCurrent)) {
            float allowedDepthDelta = depthCurrent * pushConstants.depthScale + pushConstants.depthThreshold;
            rejectDepth = !validDepth(depthHistory) || abs(depthCurrent - depthHistory) > allowedDepthDelta;
        }

        rejectHistory = rejectNormal || rejectDepth;

        if (!rejectHistory) {
            colorHistory = texture(texHistory, historyUV).rgb;
            colorHistory = clamp(colorHistory, minColor, maxColor);
        } else {
            colorHistory = colorCurrent;
            normalHistory = normalCurrent;
        }
    }

    float blendFactor = rejectHistory ? 1.0 : pushConstants.alpha;

    vec3 resultColor = mix(colorHistory, colorCurrent, blendFactor);
    vec3 resultNormal = safeNormalize(mix(normalHistory, normalCurrent, blendFactor), normalCurrent);

    fragColor = vec4(resultColor, 1.0);
    fragNormal = vec4(resultNormal, 1.0);
    fragDepth = depthCurrent;
}
