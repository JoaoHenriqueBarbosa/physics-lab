#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_texture;
uniform float u_threshold;

void main() {
    vec3 col = texture(u_texture, v_uv).rgb;
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float contrib = max(0.0, lum - u_threshold) / max(lum, 0.001);
    frag_color = vec4(col * contrib, 1.0);
}
