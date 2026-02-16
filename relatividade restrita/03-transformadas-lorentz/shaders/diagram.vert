#version 330

uniform mat4 u_model;
uniform mat4 u_viewproj;

in vec3 in_position;
in vec2 in_texcoord;

out vec2 v_uv;

void main() {
    v_uv = in_texcoord;
    gl_Position = u_viewproj * u_model * vec4(in_position, 1.0);
}
