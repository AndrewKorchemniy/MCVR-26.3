#version 460

const uint MANUAL_EXPOSURE = 1u << 0u;
const vec3 REC709_LUMINANCE = vec3(0.2126, 0.7152, 0.0722);

layout(set = 0, binding = 0) uniform sampler2D hdrTexture;

layout(std430, set = 0, binding = 2) readonly buffer ToneMappingStateBuffer {
    float exposureEv;
    float measuredLogLuminance;
    uint frameSerial;
    uint valid;
}
toneMappingState;

layout(set = 0, binding = 4, r32f) readonly uniform image2D toneCurveImage;

layout(push_constant) uniform PushConstant {
    float logLuminanceMin;
    float logLuminanceMax;
    float deltaTimeSec;
    float exposureCeiling;
    float targetLuminance;
    float meteringResponse;
    float skyMeterWeight;
    float emissiveMeterWeight;
    float manualExposureEv;
    float chromaPreservation;
    float outputContrastRange;
    float shadowDetailFloor;
    float shadowContrast;
    float shadowPivot;
    float outputOffsetEv;
    float curveLiftLimit;
    uint flags;
    float nightFactor;
}
pc;

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

bool flagSet(uint bit) {
    return (pc.flags & bit) != 0u;
}

float sampleToneCurve(float normalizedLogLuminance) {
    float texelPosition = clamp(normalizedLogLuminance, 0.0, 1.0) * 256.0 - 0.5;
    int lowerUnclamped = int(floor(texelPosition));
    int lower = clamp(lowerUnclamped, 0, 255);
    int upper = clamp(lowerUnclamped + 1, 0, 255);
    float interpolation = fract(texelPosition);
    float lowerValue = imageLoad(toneCurveImage, ivec2(lower, 0)).r;
    float upperValue = imageLoad(toneCurveImage, ivec2(upper, 0)).r;
    return mix(lowerValue, upperValue, interpolation);
}

vec3 applyChromaCorrection(vec3 color) {
    float average = dot(color, REC709_LUMINANCE);
    float correction = ((1486.4 - average * 1489.7) * average - 3.3) / ((average * 0.15 + 944.2) * average + 1.0);
    float saturationScale = 1.32 + pc.chromaPreservation * max(correction - 1.0, 0.0);
    vec3 chroma = (color - vec3(average)) * saturationScale;
    float trough = min(chroma.r, min(chroma.g, chroma.b));
    color = vec3(average) + chroma * min(1.0, average / max(-trough, 1e-5));
    float peak = max(color.r, max(color.g, color.b));
    float shoulder = max(peak - 0.5, 0.0);
    float compressedPeak = min(peak, 0.5) + shoulder / (1.0 + shoulder / 0.5);
    return color * (compressedPeak / max(peak, 1e-5));
}

vec3 toSrgb(vec3 linearColor) {
    vec3 color = max(linearColor, vec3(0.0));
    vec3 low = color * 12.92;
    vec3 high = 1.055 * pow(color, vec3(1.0 / 2.4)) - vec3(0.055);
    return mix(high, low, lessThanEqual(color, vec3(0.0031306699384003878)));
}

vec3 addOutputDither(vec3 color) {
    uint hash = (uint(abs(gl_FragCoord.x)) << 16u) + uint(abs(gl_FragCoord.y));
    hash = (hash ^ 61u) ^ (hash >> 16u);
    hash *= 9u;
    hash ^= hash >> 4u;
    hash *= 668265261u;
    hash ^= hash >> 15u;
    float noise = 0.00196078443 - float(hash) * 1.8261228e-12;
    return color + vec3(noise);
}

void main() {
    float exposureEv = toneMappingState.valid != 0u ? toneMappingState.exposureEv : 0.0;
    if (isnan(exposureEv) || isinf(exposureEv)) { exposureEv = flagSet(MANUAL_EXPOSURE) ? pc.manualExposureEv : 0.0; }

    vec3 hdrColor = texture(hdrTexture, texCoord).rgb;
    if (any(isnan(hdrColor)) || any(isinf(hdrColor))) { hdrColor = vec3(0.0); }
    vec3 exposedColor = max(hdrColor, vec3(0.0)) * exp2(clamp(exposureEv, -24.0, 24.0));
    float luminance = max(dot(exposedColor, REC709_LUMINANCE), exp2(-24.0));
    float normalizedLogLuminance =
        clamp((log2(luminance) - pc.logLuminanceMin) / max(pc.logLuminanceMax - pc.logLuminanceMin, 1e-6), 0.0, 1.0);

    float mappedLuminance = exp2(sampleToneCurve(normalizedLogLuminance));
    vec3 mappedColor = exposedColor * (mappedLuminance / luminance);
    vec3 balance = mix(vec3(0.98, 1.0, 1.03), vec3(1.04, 1.0, 0.94), smoothstep(0.02, 0.6, mappedLuminance));
    mappedColor *= balance;
    mappedColor *= mappedLuminance / max(dot(mappedColor, REC709_LUMINANCE), exp2(-24.0));
    mappedColor = applyChromaCorrection(mappedColor);
    mappedColor = toSrgb(mappedColor);
    mappedColor = addOutputDither(mappedColor);

    fragColor = vec4(clamp(mappedColor, vec3(0.0), vec3(1.0)), 1.0);
}
