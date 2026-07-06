#version 330
uniform sampler2D tex;
uniform float thickness;   // outline thickness in texels
uniform float threshold;   // edge sensitivity
in vec2 uv;
out vec4 fragColor;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    vec2 texel = (thickness / vec2(textureSize(tex, 0)));
    vec3 col = texture(tex, uv).rgb;

    // Sobel edge detection on luminance.
    float tl = luma(texture(tex, uv + texel * vec2(-1, -1)).rgb);
    float  t = luma(texture(tex, uv + texel * vec2( 0, -1)).rgb);
    float tr = luma(texture(tex, uv + texel * vec2( 1, -1)).rgb);
    float  l = luma(texture(tex, uv + texel * vec2(-1,  0)).rgb);
    float  r = luma(texture(tex, uv + texel * vec2( 1,  0)).rgb);
    float bl = luma(texture(tex, uv + texel * vec2(-1,  1)).rgb);
    float  b = luma(texture(tex, uv + texel * vec2( 0,  1)).rgb);
    float br = luma(texture(tex, uv + texel * vec2( 1,  1)).rgb);

    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
    float edge = sqrt(gx*gx + gy*gy);

    // Ink the edges black over the source color.
    float ink = step(threshold, edge);
    fragColor = vec4(col * (1.0 - ink), 1.0);
}
