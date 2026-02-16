#version 330

uniform mat4 u_model;
uniform mat4 u_viewproj;

in vec3 in_position;

out vec3 v_world;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world = world.xyz;
    gl_Position = u_viewproj * world;
}
