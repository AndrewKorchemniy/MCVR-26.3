#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(std140, set = 1, binding = 1) uniform PostUniform {
    OverlayPostUBO ubo;
};

layout(location = 0) out vec2 texCoord;
layout(location = 1) out vec2 sampleStep;

void main() {

    vec2 positions[3] = vec2[](vec2(-1.0, -1.0),
                               vec2(-1.0, 3.0),
                               vec2(3.0, -1.0)
    );

    vec2 texCoords[3] = vec2[](vec2(0.0, 0.0),
                               vec2(0.0, 2.0),
                               vec2(2.0, 0.0)
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);

    vec2 oneTexel = 1.0 / ubo.inSize;
    sampleStep = oneTexel * ubo.blurDir;

    texCoord = texCoords[gl_VertexIndex].xy;
}
