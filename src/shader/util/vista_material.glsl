#ifndef VISTA_MATERIAL_GLSL
#define VISTA_MATERIAL_GLSL

vec4 sampleVistaTile(sampler2D atlas, uint tile, vec2 uv, float level) {
    float size = exp2(4.0 - level);
    vec2 origin = vec2(tile % 256u, tile / 256u) * size;
    vec2 pixel = origin + clamp(fract(uv) * size, vec2(0.5), vec2(size - 0.5));
    return textureLod(atlas, pixel / vec2(textureSize(atlas, int(level))), level);
}

vec3 vistaAlbedo(sampler2D atlas, MaterialVertex vertex, vec2 uv, vec3 color, float coneWidth) {
    uint tile = vertex.textureID & 65535u;
    if (tile == 0u) { return color; }
    float level = clamp(log2(max(coneWidth * 16.0, 1.0)), 0.0, 4.0);
    vec4 low = sampleVistaTile(atlas, tile, uv, floor(level));
    vec4 high = sampleVistaTile(atlas, tile, uv, ceil(level));
    vec4 ratio = mix(low, high, fract(level));
    return mix(color, clamp(color * ratio.rgb * 2.0, 0.0, 1.0), ratio.a);
}

// The tile's alpha is texture coverage (0 for cut-out texels). Surfaces multiply it into their alpha, which only has an
// effect in the cut-out and transparent alpha modes since the opaque mode ignores alpha.
float vistaAlpha(sampler2D atlas, MaterialVertex vertex, vec2 uv, float coneWidth) {
    uint tile = vertex.textureID & 65535u;
    if (tile == 0u) { return 1.0; }
    float level = clamp(log2(max(coneWidth * 16.0, 1.0)), 0.0, 4.0);
    float low = sampleVistaTile(atlas, tile, uv, floor(level)).a;
    float high = sampleVistaTile(atlas, tile, uv, ceil(level)).a;
    return mix(low, high, fract(level));
}

#endif
