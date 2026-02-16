#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform float u_beta;       // v/c
uniform float u_gamma;
uniform float u_range;       // visible ±range
uniform float u_time;

// events (x, ct) in frame S — up to 4
uniform int   u_num_events;
uniform vec2  u_ev0;
uniform vec2  u_ev1;
uniform vec2  u_ev2;
uniform vec2  u_ev3;

// simultaneity pair indices (-1 = none)
uniform int   u_simul_a;
uniform int   u_simul_b;

// ── helpers ─────────────────────────────────────────────────
float grid_aa(float coord) {
    float d = abs(fract(coord + 0.5) - 0.5);
    return 1.0 - smoothstep(0.0, fwidth(coord) * 1.5, d);
}

float axis_aa(float coord) {
    return 1.0 - smoothstep(0.0, fwidth(coord) * 2.5, abs(coord));
}

float line_h(float ct_val, float ct_here, float x_here, float x_min, float x_max) {
    float on = smoothstep(fwidth(ct_here) * 2.0, 0.0, abs(ct_here - ct_val));
    float inx = step(x_min, x_here) * step(x_here, x_max);
    // dashed
    float dash = step(0.4, fract(x_here * 2.0));
    return on * inx * dash;
}

void main() {
    float R = u_range;
    vec2 st = (v_uv * 2.0 - 1.0) * R;   // (x, ct)
    float x  = st.x;
    float ct = st.y;

    vec3 col = vec3(0.012, 0.012, 0.03);

    // ── causal regions (faint tint) ─────────────────────────
    float ct2 = ct * ct;
    float x2  = x * x;
    if (ct2 > x2) {
        float f = ct > 0.0 ? 0.025 : 0.015;
        col += vec3(f, f, 0.005);
    }

    // ── frame S grid (cool blue, orthogonal) ────────────────
    float sg = max(grid_aa(x), grid_aa(ct));
    col += vec3(0.0, 0.18, 0.32) * sg * 0.28;

    // major grid every 2 units
    float sgm = max(grid_aa(x / 2.0), grid_aa(ct / 2.0));
    col += vec3(0.0, 0.22, 0.4) * sgm * 0.15;

    // ── frame S' grid (warm amber, Lorentz-skewed) ──────────
    float xp  = u_gamma * (x  - u_beta * ct);
    float ctp = u_gamma * (ct - u_beta * x);

    float spg = max(grid_aa(xp), grid_aa(ctp));
    col += vec3(0.35, 0.17, 0.0) * spg * 0.32;

    float spgm = max(grid_aa(xp / 2.0), grid_aa(ctp / 2.0));
    col += vec3(0.4, 0.2, 0.0) * spgm * 0.15;

    // ── S axes ──────────────────────────────────────────────
    col += vec3(0.0, 0.55, 1.0) * axis_aa(x)  * 0.65;  // ct axis
    col += vec3(0.0, 0.55, 1.0) * axis_aa(ct) * 0.65;  // x  axis

    // ── S' axes ─────────────────────────────────────────────
    col += vec3(1.0, 0.6, 0.0) * axis_aa(xp)  * 0.6;   // ct' axis
    col += vec3(1.0, 0.6, 0.0) * axis_aa(ctp) * 0.6;   // x'  axis

    // ── light cone ──────────────────────────────────────────
    float lc1 = abs(x - ct);
    float lc2 = abs(x + ct);
    float lc_d = min(lc1, lc2);
    float fw = fwidth(x);
    col += vec3(1.0, 0.92, 0.25) * (1.0 - smoothstep(0.0, fw * 2.5, lc_d)) * 0.55;
    col += vec3(1.0, 0.85, 0.15) * (1.0 - smoothstep(0.0, fw * 18.0, lc_d)) * 0.12;

    // ── simultaneity line (dashed, green) ───────────────────
    vec2 evs[4];
    evs[0] = u_ev0; evs[1] = u_ev1; evs[2] = u_ev2; evs[3] = u_ev3;

    if (u_simul_a >= 0 && u_simul_b >= 0) {
        vec2 ea = evs[u_simul_a];
        vec2 eb = evs[u_simul_b];
        float mn_x = min(ea.x, eb.x);
        float mx_x = max(ea.x, eb.x);
        float sl = line_h(ea.y, ct, x, mn_x, mx_x);
        col += vec3(0.2, 0.9, 0.4) * sl * 0.45;
    }

    // ── events ──────────────────────────────────────────────
    vec3 ev_col[4];
    ev_col[0] = vec3(0.1, 1.0, 0.45);   // green
    ev_col[1] = vec3(1.0, 0.22, 0.28);  // red
    ev_col[2] = vec3(0.35, 0.55, 1.0);  // blue
    ev_col[3] = vec3(0.95, 0.35, 0.95); // magenta

    for (int i = 0; i < 4; i++) {
        if (i >= u_num_events) break;
        float d = length(st - evs[i]);
        float pulse = 0.85 + 0.15 * sin(u_time * 2.5 + float(i) * 1.5);
        col += ev_col[i] * smoothstep(R * 0.022, 0.0, d) * 3.5 * pulse;
        col += ev_col[i] * smoothstep(R * 0.06, 0.0, d) * 0.7;
    }

    // ── border fade ─────────────────────────────────────────
    float edge = max(abs(v_uv.x - 0.5), abs(v_uv.y - 0.5)) * 2.0;
    col *= smoothstep(1.0, 0.92, edge);

    frag_color = vec4(col, smoothstep(1.0, 0.95, edge));
}
