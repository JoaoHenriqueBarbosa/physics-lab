#version 330

uniform mat4  u_model;
uniform mat4  u_viewproj;
uniform float u_contraction;   // 1/γ = √(1 − v²/c²)
uniform vec3  u_motion_dir;    // unit vector along direction of motion

in vec3 in_position;
in vec3 in_normal;

out vec3 v_world;
out vec3 v_normal;

void main() {
    // ── Lorentz contraction in local space ──
    vec3 pos = in_position;
    float proj = dot(pos, u_motion_dir);
    pos -= u_motion_dir * proj * (1.0 - u_contraction);

    vec4 world = u_model * vec4(pos, 1.0);
    v_world = world.xyz;

    // normal transform: inverse-transpose of the contraction
    vec3 n = in_normal;
    float n_proj = dot(n, u_motion_dir);
    n += u_motion_dir * n_proj * (1.0 / max(u_contraction, 0.001) - 1.0);
    v_normal = normalize(mat3(u_model) * n);

    gl_Position = u_viewproj * world;
}
