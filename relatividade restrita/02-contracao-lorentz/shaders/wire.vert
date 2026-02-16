#version 330

uniform mat4  u_model;
uniform mat4  u_viewproj;
uniform float u_contraction;
uniform vec3  u_motion_dir;

in vec3 in_position;

void main() {
    vec3 pos = in_position;
    float proj = dot(pos, u_motion_dir);
    pos -= u_motion_dir * proj * (1.0 - u_contraction);
    gl_Position = u_viewproj * u_model * vec4(pos, 1.0);
}
