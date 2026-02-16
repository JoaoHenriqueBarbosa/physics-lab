#version 330

in vec3 v_world;
out vec4 frag_color;

void main() {
    vec2 c = v_world.xz;

    // minor grid lines (every 1 unit)
    vec2 g = abs(fract(c - 0.5) - 0.5) / fwidth(c);
    float minor = 1.0 - min(min(g.x, g.y), 1.0);

    // major grid lines (every 5 units)
    vec2 g5 = abs(fract(c / 5.0 - 0.5) - 0.5) / fwidth(c / 5.0);
    float major = 1.0 - min(min(g5.x, g5.y), 1.0);

    // axis lines
    float ax = smoothstep(0.06, 0.0, abs(c.x)) * 0.8;
    float az = smoothstep(0.06, 0.0, abs(c.y)) * 0.8;

    // distance fade
    float fade = exp(-length(c) * 0.022);

    vec3 col = vec3(0.0, 0.45, 0.65) * minor * 0.25
             + vec3(0.0, 0.7, 1.0) * major * 0.55
             + vec3(0.2, 0.5, 1.0) * ax
             + vec3(1.0, 0.3, 0.2) * az;

    float alpha = max(minor * 0.35, major * 0.7) * fade;
    alpha = max(alpha, (ax + az) * fade);

    frag_color = vec4(col * fade, alpha);
}
