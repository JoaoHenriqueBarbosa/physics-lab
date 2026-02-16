#version 330
/*
 * Lente Gravitacional de Schwarzschild — geodésicas exatas
 *
 * Resolve a equação de órbita de fótons na métrica de Schwarzschild:
 *     d²u/dφ² + u = 3Mu²       (u = 1/r, unidades geométricas G=c=1)
 *
 * Integração via Runge-Kutta de 4ª ordem (RK4).
 *
 * Referências:
 *   - Schwarzschild, K. (1916) — métrica original
 *   - Luminet, J.-P. (1979) — Image of a Spherical Black Hole with Thin
 *     Accretion Disk, Astron. Astrophys. 75, 228–235
 *   - Bruneton, E. (2020) — arXiv:2010.08735
 */

in  vec2 v_uv;
out vec4 frag_color;

/* ── uniforms ─────────────────────────────────────────── */
uniform vec3  u_cam_pos;
uniform vec3  u_cam_fwd;
uniform vec3  u_cam_right;
uniform vec3  u_cam_up;
uniform float u_fov;           // half-angle, radians
uniform float u_aspect;

uniform float u_M;             // massa (rs = 2M)
uniform float u_disk_inner;    // raio interno do disco (ISCO = 6M)
uniform float u_disk_outer;    // raio externo
uniform float u_temp_scale;    // escala de temperatura (K) para visibilidade
uniform float u_time;

const float PI        = 3.14159265358979;
const int   MAX_STEPS = 800;
const float DPHI      = 0.008;
const float ESCAPE_R  = 300.0;

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Blackbody  (Planck → sRGB, aproximação de Mitchell Charity)
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
vec3 blackbody(float T) {
    // T em Kelvin.  Válido ~1000 K – 40000 K
    float t = T / 1000.0;
    if (t < 0.4) return vec3(0.0);

    float r, g, b;

    // vermelho
    if      (t < 6.6)  r = 1.0;
    else                r = clamp(1.292936 * pow(t - 0.6, -0.1332047), 0.0, 1.0);

    // verde
    if      (t < 0.9)  g = 0.0;
    else if (t < 6.6)  g = clamp(0.390082 * log(t) - 0.631841, 0.0, 1.0);
    else                g = clamp(1.129891 * pow(t - 0.6, -0.0755148), 0.0, 1.0);

    // azul
    if      (t < 2.0)  b = 0.0;
    else if (t < 6.6)  b = clamp(0.543206 * log(t - 1.0) - 1.19625, 0.0, 1.0);
    else                b = 1.0;

    return vec3(r, g, b);
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Campo estelar procedural  (3D direction → color)
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
float hash31(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

vec3 starfield(vec3 dir) {
    vec3 col = vec3(0.0);
    dir = normalize(dir);

    for (int layer = 0; layer < 3; layer++) {
        float scale = 80.0 + float(layer) * 60.0;
        vec3 grid  = floor(dir * scale);
        vec3 local = fract(dir * scale) - 0.5;
        float h    = hash31(grid + float(layer) * 100.0);

        if (h > 0.97) {
            float bright = (h - 0.97) / 0.03;
            float d      = length(local);
            float glow   = exp(-d * d * 60.0) * bright;

            // cor estelar (temperatura aleatória)
            float star_t = 3000.0 + h * 25000.0;
            col += blackbody(star_t) * glow * 3.0;
        }
    }
    // leve fundo nebular
    float neb = hash31(floor(dir * 15.0)) * 0.012;
    col += vec3(0.05, 0.03, 0.08) * neb;

    return col;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Perfil de temperatura do disco
 *
 * Modelo thin-disk (Novikov-Thorne simplificado):
 *   T(r) ∝ [ f(r) / r³ ]^(1/4)
 *   f(r) = 1 − √(r_isco / r)
 *
 * Pico em r ≈ 1.36 × r_isco.
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
float disk_temperature(float r, float M) {
    float r_isco = 6.0 * M;
    if (r <= r_isco) return 0.0;
    float f = 1.0 - sqrt(r_isco / r);
    return pow(max(f, 0.0), 0.25) * pow(r_isco / r, 0.75);
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Disco de acreção — cor final com redshift
 *
 * Fator de redshift total (gravitacional + Doppler):
 *   g = √(1 − 3M/r) / (1 − L_z Ω)
 *
 * onde  Ω = √(M/r³)  é a velocidade angular Kepleriana
 * e     L_z = b · (n_orbit · ŷ)  é a componente z do momento
 *       angular do fóton.
 *
 * Intensidade observada:  I_obs ∝ g⁴ × B(g·T_emit)
 * (fator g³ para intensidade específica × g para shift de frequência)
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
vec3 disk_color(float r, float M, float Lz) {
    float T_norm = disk_temperature(r, M);
    if (T_norm < 1e-6) return vec3(0.0);

    float T_emit = T_norm * u_temp_scale;

    // redshift gravitacional + Doppler
    float Omega = sqrt(M / (r * r * r));
    float denom = 1.0 - Lz * Omega;
    float g     = sqrt(max(1.0 - 3.0 * M / r, 0.0)) / max(denom, 0.01);
    g = clamp(g, 0.05, 6.0);

    float T_obs = T_emit * g;
    vec3  col   = blackbody(T_obs);

    // intensidade ∝ g⁴  (invariância relativística)
    col *= pow(g, 4.0);

    // brilho radial (proporcional ao fluxo emitido F ∝ T⁴)
    // escalamos para HDR — bloom cuida do tone mapping
    float flux = T_norm * T_norm * T_norm * T_norm;
    col *= flux * 80.0;

    return col;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Geodésica — RK4 na equação de órbita de Schwarzschild
 *
 * Sistema:
 *   du/dφ = p
 *   dp/dφ = −u + 3Mu²
 *
 * Condições iniciais determinadas pelo parâmetro de
 * impacto b = r₀ sin α / √(1 − rs/r₀).
 *
 * A cada passo verificamos:
 *   1) Horizonte de eventos  (r ≤ rs)
 *   2) Cruzamento do disco   (y muda de sinal)
 *   3) Escape                (r > R_escape e dr/dφ > 0)
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_aspect;

    float fov_scale = tan(u_fov);
    vec3 ray_dir = normalize(
        u_cam_fwd
        + uv.x * fov_scale * u_cam_right
        + uv.y * fov_scale * u_cam_up
    );

    float M  = u_M;
    float rs = 2.0 * M;
    float r0 = length(u_cam_pos);

    /* ── decomposição radial / tangencial ─────────────── */
    vec3  r_hat = u_cam_pos / r0;
    float cos_a = dot(ray_dir, r_hat);
    vec3  tang  = ray_dir - cos_a * r_hat;
    float sin_a = length(tang);

    // raio quase radial → resultado trivial
    if (sin_a < 1e-7) {
        if (cos_a < 0.0)
            frag_color = vec4(0.0, 0.0, 0.0, 1.0);   // direto no BH
        else
            frag_color = vec4(starfield(ray_dir), 1.0);
        return;
    }

    vec3 t_hat = tang / sin_a;

    /* ── plano orbital ────────────────────────────────── */
    vec3 e1 = r_hat;          // direção radial (câmera → BH)
    vec3 e2 = t_hat;          // direção tangencial
    vec3 n_orbit = cross(e1, e2);  // normal ao plano orbital

    /* ── parâmetro de impacto ─────────────────────────── */
    float b = r0 * sin_a / sqrt(max(1.0 - rs / r0, 1e-8));

    /* ── L_z para Doppler  (disco no plano XZ, rotação em +y) */
    float Lz = b * n_orbit.y;

    /* ── condições iniciais ───────────────────────────── */
    float u = 1.0 / r0;
    float p_sq = 1.0 / (b * b) - u * u + 2.0 * M * u * u * u;
    float p = sqrt(max(p_sq, 0.0));
    if (cos_a > 0.0) p = -p;   // raio apontando para fora → u decrescente

    /* ── integração ───────────────────────────────────── */
    float phi = 0.0;
    vec3  color = vec3(0.0);
    float acc_alpha = 0.0;

    // y anterior para detectar cruzamento do disco
    float prev_y = r0 * (1.0 * e1.y + 0.0 * e2.y);  // pos em phi=0

    for (int i = 0; i < MAX_STEPS; i++) {
        /* ── RK4 ──────────────────────────────────────── */
        float k1u = p;
        float k1p = -u + 3.0 * M * u * u;

        float hu = u + 0.5 * DPHI * k1u;
        float hp = p + 0.5 * DPHI * k1p;
        float k2u = hp;
        float k2p = -hu + 3.0 * M * hu * hu;

        hu = u + 0.5 * DPHI * k2u;
        hp = p + 0.5 * DPHI * k2p;
        float k3u = hp;
        float k3p = -hu + 3.0 * M * hu * hu;

        hu = u + DPHI * k3u;
        hp = p + DPHI * k3p;
        float k4u = hp;
        float k4p = -hu + 3.0 * M * hu * hu;

        u   += DPHI * (k1u + 2.0*k2u + 2.0*k3u + k4u) / 6.0;
        p   += DPHI * (k1p + 2.0*k2p + 2.0*k3p + k4p) / 6.0;
        phi += DPHI;

        float r = 1.0 / max(u, 1e-10);

        /* ── horizonte de eventos ─────────────────────── */
        if (r <= rs * 1.005) {
            // fóton capturado — preto absoluto
            break;
        }

        /* ── posição 3D no plano orbital ──────────────── */
        float cp = cos(phi);
        float sp = sin(phi);
        vec3  pos = r * (cp * e1 + sp * e2);
        float cur_y = pos.y;

        /* ── cruzamento do disco de acreção ───────────── */
        if (prev_y * cur_y < 0.0 && acc_alpha < 0.995) {
            // interpolação linear para ponto exato
            float frac = abs(prev_y) / (abs(prev_y) + abs(cur_y) + 1e-12);

            // raio no ponto de cruzamento (aproximação linear)
            float phi_cross = phi - DPHI * (1.0 - frac);
            float cp2 = cos(phi_cross);
            float sp2 = sin(phi_cross);
            // u interpolado
            float u_prev = 1.0 / max(length(
                (1.0/max(u - DPHI*(k1u+2.0*k2u+2.0*k3u+k4u)/6.0, 1e-10))
                * (cos(phi - DPHI) * e1 + sin(phi - DPHI) * e2)), 1e-10);
            // simplificação: usar r atual
            float r_cross = length(r * (cp * e1 + sp * e2)
                                   - (r * cur_y / (cur_y - prev_y + 1e-12))
                                     * (cp * e1 + sp * e2 - (r / max(1.0/u,r)) * (cos(phi-DPHI)*e1 + sin(phi-DPHI)*e2)));
            // abordagem mais simples e robusta:
            r_cross = r;  // r no passo atual (boa aproximação para dphi pequeno)

            if (r_cross >= u_disk_inner && r_cross <= u_disk_outer) {
                vec3 dcol = disk_color(r_cross, M, Lz);

                // opacidade diminui com r (disco mais fino nas bordas)
                float opacity = 0.92 * smoothstep(u_disk_outer, u_disk_inner, r_cross);
                opacity = max(opacity, 0.3);

                color     += dcol * opacity * (1.0 - acc_alpha);
                acc_alpha += opacity * (1.0 - acc_alpha);
            }
        }
        prev_y = cur_y;

        /* ── escape ───────────────────────────────────── */
        if (r > ESCAPE_R && p < 0.0) {
            // direção de escape: tangente à trajetória
            float v_r   = -p / (u * u);   // dr/dφ
            float v_phi = 1.0 / u;         // r  (componente tangencial)
            vec3  d_r   = cp * e1 + sp * e2;     // direção radial local
            vec3  d_t   = -sp * e1 + cp * e2;    // direção tangencial local
            vec3  d_out = normalize(v_r * d_r + v_phi * d_t);

            vec3 bg = starfield(d_out);
            color     += bg * (1.0 - acc_alpha);
            acc_alpha  = 1.0;
            break;
        }
    }

    frag_color = vec4(color, 1.0);
}
