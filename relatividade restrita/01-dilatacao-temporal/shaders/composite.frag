#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform sampler2D u_hud;
uniform float u_bloom_strength;
uniform int   u_bloom_on;

void main() {
    // chromatic aberration
    float ca = 0.0015;
    vec3 col;
    col.r = texture(u_scene, v_uv + vec2( ca, 0.0)).r;
    col.g = texture(u_scene, v_uv).g;
    col.b = texture(u_scene, v_uv + vec2(-ca, 0.0)).b;

    // bloom
    if (u_bloom_on != 0) {
        vec3 bloom = texture(u_bloom, v_uv).rgb;
        col += bloom * u_bloom_strength;
    }

    // vignette
    float d = length(v_uv - 0.5);
    col *= 1.0 - 0.45 * d * d;

    // tone mapping (simple reinhard)
    col = col / (col + 1.0);

    // HUD overlay
    vec4 hud = texture(u_hud, v_uv);
    col = mix(col, hud.rgb, hud.a);

    frag_color = vec4(col, 1.0);
}
