#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform float u_time;   // clock time in seconds
uniform vec3  u_color;  // accent color

// signed distance to a line segment
float sdSeg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    vec2 p = v_uv * 2.0 - 1.0;
    float r = length(p);

    if (r > 1.05) discard;

    // ── face ──────────────────────────────────────
    vec3 col = mix(vec3(0.025, 0.025, 0.06), vec3(0.01), r);

    // outer ring (double)
    col += u_color * 1.6 * smoothstep(0.018, 0.0, abs(r - 0.94));
    col += u_color * 0.4 * smoothstep(0.012, 0.0, abs(r - 0.90));

    // ── hour ticks ────────────────────────────────
    for (int i = 0; i < 12; i++) {
        float a = float(i) * 6.2831853 / 12.0;
        vec2 d = vec2(sin(a), cos(a));
        float dist = sdSeg(p, d * 0.76, d * 0.87);
        col += u_color * smoothstep(0.014, 0.003, dist);
    }

    // ── minute ticks ──────────────────────────────
    for (int i = 0; i < 60; i++) {
        float a = float(i) * 6.2831853 / 60.0;
        vec2 d = vec2(sin(a), cos(a));
        float dist = sdSeg(p, d * 0.83, d * 0.87);
        col += u_color * 0.22 * smoothstep(0.006, 0.001, dist);
    }

    // ── hands ─────────────────────────────────────
    float sec = mod(u_time, 60.0);
    float mn  = mod(u_time / 60.0, 60.0);
    float hr  = mod(u_time / 3600.0, 12.0);

    // hour hand
    float ha = hr * 6.2831853 / 12.0;
    vec2 hd = vec2(sin(ha), cos(ha));
    col += u_color * smoothstep(0.024, 0.009, sdSeg(p, vec2(0.0), hd * 0.46));

    // minute hand
    float ma = mn * 6.2831853 / 60.0;
    vec2 md = vec2(sin(ma), cos(ma));
    col += u_color * smoothstep(0.014, 0.004, sdSeg(p, vec2(0.0), md * 0.66));

    // second hand – red/orange with glow
    float sa = sec * 6.2831853 / 60.0;
    vec2 sd = vec2(sin(sa), cos(sa));
    float sd_d = sdSeg(p, vec2(0.0), sd * 0.80);
    col += vec3(1.0, 0.18, 0.06) * smoothstep(0.008, 0.0, sd_d);
    col += vec3(1.0, 0.10, 0.02) * smoothstep(0.05, 0.0, sd_d) * 0.45;

    // center dot
    col += u_color * smoothstep(0.04, 0.018, r);

    // ── edge fade + outer glow ────────────────────
    float alpha = smoothstep(1.05, 0.94, r);
    col += u_color * 0.15 * smoothstep(1.08, 0.96, r) * (1.0 - smoothstep(0.94, 0.96, r));

    frag_color = vec4(col, alpha);
}
