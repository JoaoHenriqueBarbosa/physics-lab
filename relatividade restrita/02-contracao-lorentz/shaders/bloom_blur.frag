#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_texture;
uniform vec2 u_direction;   // (1/w, 0) or (0, 1/h)

const float W0 = 0.227027;
const float W1 = 0.194595;
const float W2 = 0.121622;
const float W3 = 0.054054;
const float W4 = 0.016216;

void main() {
    vec3 result = texture(u_texture, v_uv).rgb * W0;

    result += texture(u_texture, v_uv + u_direction * 1.0).rgb * W1;
    result += texture(u_texture, v_uv - u_direction * 1.0).rgb * W1;

    result += texture(u_texture, v_uv + u_direction * 2.0).rgb * W2;
    result += texture(u_texture, v_uv - u_direction * 2.0).rgb * W2;

    result += texture(u_texture, v_uv + u_direction * 3.0).rgb * W3;
    result += texture(u_texture, v_uv - u_direction * 3.0).rgb * W3;

    result += texture(u_texture, v_uv + u_direction * 4.0).rgb * W4;
    result += texture(u_texture, v_uv - u_direction * 4.0).rgb * W4;

    frag_color = vec4(result, 1.0);
}
