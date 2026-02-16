#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform float u_time;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec3 col = vec3(0.004, 0.004, 0.018);

    // star layers at different densities
    for (int layer = 0; layer < 3; layer++) {
        float scale = 200.0 + float(layer) * 150.0;
        vec2 grid = floor(v_uv * scale);
        float h = hash(grid + float(layer) * 73.0);

        if (h > 0.985) {
            vec2 center = (grid + 0.5) / scale;
            float d = length(v_uv - center) * scale;
            float twinkle = 0.6 + 0.4 * sin(u_time * (h * 4.0 + 0.5) + h * 50.0);
            float brightness = smoothstep(1.2, 0.0, d) * twinkle;

            // slight color variation
            vec3 star_col = mix(vec3(0.8, 0.85, 1.0), vec3(1.0, 0.9, 0.7), h * 7.0 - 6.0);
            col += star_col * brightness * (0.4 + float(layer) * 0.3);
        }
    }

    frag_color = vec4(col, 1.0);
}
