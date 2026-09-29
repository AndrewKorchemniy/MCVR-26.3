#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 1) uniform sampler2D frame;
layout(std140, set = 1, binding = 1) uniform PostUniform {
    OverlayPostUBO ubo;
};

layout(location = 0) in vec2 texCoord;
layout(location = 1) in vec2 sampleStep;

layout(location = 0) out vec4 fragColor;

void main() {
    vec4 blurred = vec4(0.0);
    float actualRadius = round(ubo.radius * ubo.radiusMultiplier);
    for (float a = -actualRadius + 0.5; a <= actualRadius; a += 2.0) {
        blurred += texture(frame, texCoord + sampleStep * a);
    }
    blurred += texture(frame, texCoord + sampleStep * actualRadius) / 2.0;
    fragColor = blurred / (actualRadius + 0.5);
}
