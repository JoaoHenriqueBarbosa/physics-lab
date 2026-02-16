#version 330

in vec3 v_world;
in vec3 v_normal;

out vec4 frag_color;

uniform vec3  u_color;
uniform vec3  u_cam_pos;
uniform float u_alpha;

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_cam_pos - v_world);
    vec3 L = normalize(vec3(0.4, 1.0, 0.6));

    // Phong
    float amb  = 0.12;
    float diff = max(dot(N, L), 0.0) * 0.55;
    vec3  R    = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 48.0) * 0.5;

    // Fresnel edge glow
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);

    vec3 col = u_color * (amb + diff)
             + vec3(1.0) * spec
             + u_color * fresnel * 2.0;

    float alpha = u_alpha + fresnel * 0.45;

    frag_color = vec4(col, alpha);
}
