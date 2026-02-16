#version 330

out vec4 frag_color;

uniform vec3  u_color;
uniform float u_alpha;

void main() {
    frag_color = vec4(u_color, u_alpha);
}
