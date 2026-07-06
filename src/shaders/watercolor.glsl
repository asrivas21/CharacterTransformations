#version 330
uniform sampler2D tex;
uniform float time;
in vec2 uv;
out vec4 fragColor;

// Simple value-noise for paper texture + edge bleeding.
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(hash(i), hash(i+vec2(1,0)), u.x),
               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), u.x), u.y);
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(tex, 0));
    // Slight UV distortion -> color bleeding.
    vec2 warp = vec2(noise(uv*40.0), noise(uv*40.0+7.3)) - 0.5;
    vec3 col = texture(tex, uv + warp * texel * 3.0).rgb;

    // Posterize gently for flat wash regions.
    col = floor(col * 6.0) / 6.0;

    // Warm desaturation (ATLA palette).
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(lum), col, 0.65);
    col *= vec3(1.08, 1.0, 0.9);   // warm bias

    // Paper grain.
    float grain = noise(uv * 800.0) * 0.06;
    col += grain - 0.03;

    fragColor = vec4(col, 1.0);
}
