#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform vec3 u_color;

void main() {
    vec2 p = v_uv * 2.0 - 1.0;
    float r = length(p);
    if (r > 1.0) discard;

    float glow = pow(1.0 - r, 1.5);
    float ring = smoothstep(0.03, 0.0, abs(r - 0.82)) * 1.8;
    float inner = smoothstep(0.4, 0.0, r) * 0.3;

    vec3 col = u_color * (glow + ring + inner);
    float alpha = glow * 0.85 + ring;

    frag_color = vec4(col, alpha);
}
