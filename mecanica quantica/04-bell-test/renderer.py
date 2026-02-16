"""
ModernGL renderer — 4-panel Bell test visualization.

Uses EGL standalone context → FBO 1920×1080 → fbo.read() → PIL text → ffmpeg.
All shaders are GLSL 3.30, embedded as strings.
Matrix math done with NumPy (no pyrr).
"""

import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont

# ── Constants ────────────────────────────────────────────

W, H = 1920, 1080
PW, PH = W // 2, H // 2  # panel size: 960×540

BG = (0.055, 0.055, 0.055, 1.0)  # #0e0e0e

COL_CYAN     = (0.0, 0.898, 1.0)      # #00e5ff
COL_RED      = (1.0, 0.353, 0.353)     # #ff5a5a
COL_GREEN    = (0.298, 0.686, 0.314)   # #4caf50
COL_GREY     = (0.4, 0.4, 0.4)        # #666666
COL_YELLOW   = (1.0, 0.843, 0.251)    # #ffd740
COL_WHITE    = (1.0, 1.0, 1.0)

try:
    _F22 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
    _F18 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
    _F14 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
except OSError:
    _F22 = _F18 = _F14 = ImageFont.load_default()


# ── Matrix helpers (pure NumPy) ──────────────────────────

def _perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _ortho(l, r, b, t, n, f):
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = 2 / (r - l)
    m[1, 1] = 2 / (t - b)
    m[2, 2] = -2 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    m[3, 3] = 1.0
    return m


def _look_at(eye, target, up):
    eye, target, up = [np.asarray(v, dtype="f4") for v in (eye, target, up)]
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype="f4")
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype="f4")
    m[0, 0] = c; m[0, 2] = s
    m[2, 0] = -s; m[2, 2] = c
    return m


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype="f4")
    m[0, 0] = c; m[0, 1] = -s
    m[1, 0] = s; m[1, 1] = c
    return m


def _translate(tx, ty, tz):
    m = np.eye(4, dtype="f4")
    m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m


def _scale(sx, sy, sz):
    m = np.eye(4, dtype="f4")
    m[0, 0] = sx; m[1, 1] = sy; m[2, 2] = sz
    return m


# ── Geometry builders ────────────────────────────────────

def _uv_sphere(n_lat=24, n_lon=48):
    """Returns (vertices, normals, indices) for a unit sphere."""
    verts = []
    norms = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon + 1):
            phi = 2 * np.pi * j / n_lon
            x = np.sin(theta) * np.cos(phi)
            y = np.cos(theta)
            z = np.sin(theta) * np.sin(phi)
            verts.append((x, y, z))
            norms.append((x, y, z))
    verts = np.array(verts, dtype="f4")
    norms = np.array(norms, dtype="f4")

    indices = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * (n_lon + 1) + j
            b = a + n_lon + 1
            indices.extend([a, b, a + 1, a + 1, b, b + 1])
    indices = np.array(indices, dtype="i4")
    return verts, norms, indices


def _cylinder(radius=0.04, height=1.0, segments=12):
    """Cylinder along Y axis, origin at base."""
    verts = []
    norms = []
    indices = []
    for i in range(segments + 1):
        angle = 2 * np.pi * i / segments
        x, z = radius * np.cos(angle), radius * np.sin(angle)
        nx, nz = np.cos(angle), np.sin(angle)
        verts.append((x, 0, z))
        norms.append((nx, 0, nz))
        verts.append((x, height, z))
        norms.append((nx, 0, nz))
    for i in range(segments):
        a = i * 2
        indices.extend([a, a + 1, a + 2, a + 2, a + 1, a + 3])
    return np.array(verts, dtype="f4"), np.array(norms, dtype="f4"), np.array(indices, dtype="i4")


def _cone(radius=0.1, height=0.2, segments=12):
    """Cone along Y, base at y=0, tip at y=height."""
    verts = []
    norms = []
    indices = []
    # Base ring + tip
    slope = radius / height
    for i in range(segments + 1):
        angle = 2 * np.pi * i / segments
        x, z = radius * np.cos(angle), radius * np.sin(angle)
        ny = slope
        ln = np.sqrt(1 + ny * ny)
        norms.append((np.cos(angle) / ln, ny / ln, np.sin(angle) / ln))
        verts.append((x, 0, z))
    tip_idx = len(verts)
    verts.append((0, height, 0))
    norms.append((0, 1, 0))
    for i in range(segments):
        indices.extend([i, tip_idx, i + 1])
    return np.array(verts, dtype="f4"), np.array(norms, dtype="f4"), np.array(indices, dtype="i4")


def _circle_outline(n=64, radius=1.0):
    """Circle line-loop vertices on XY plane."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.zeros((n, 3), dtype="f4")
    verts[:, 0] = radius * np.cos(angles)
    verts[:, 1] = radius * np.sin(angles)
    return verts


# ── Shaders ──────────────────────────────────────────────

_VERT_3D = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_position;
in vec3 in_normal;
out vec3 v_normal;
out vec3 v_frag_pos;
void main() {
    vec4 world = model * vec4(in_position, 1.0);
    v_frag_pos = world.xyz;
    v_normal = mat3(transpose(inverse(model))) * in_normal;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

_FRAG_PHONG = """
#version 330
uniform vec3 color;
uniform float alpha;
uniform vec3 light_dir;
in vec3 v_normal;
in vec3 v_frag_pos;
out vec4 frag_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(light_dir);
    float diff = max(dot(n, l), 0.0);
    float amb = 0.15;
    float spec = pow(max(dot(reflect(-l, n), normalize(-v_frag_pos)), 0.0), 32.0) * 0.4;
    vec3 c = color * (amb + diff * 0.7) + vec3(1.0) * spec;
    frag_color = vec4(c, alpha);
}
"""

_VERT_2D = """
#version 330
uniform mat4 mvp;
in vec2 in_position;
void main() {
    gl_Position = mvp * vec4(in_position, 0.0, 1.0);
}
"""

_FRAG_FLAT = """
#version 330
uniform vec4 color;
out vec4 frag_color;
void main() {
    frag_color = color;
}
"""

_VERT_POINT = """
#version 330
uniform mat4 mvp;
uniform float point_size;
in vec2 in_position;
in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 0.0, 1.0);
    gl_PointSize = point_size;
    v_color = in_color;
}
"""

_FRAG_POINT = """
#version 330
in vec3 v_color;
out vec4 frag_color;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d = dot(p, p);
    if (d > 1.0) discard;
    float a = exp(-d * 3.0);
    frag_color = vec4(v_color, a);
}
"""

_VERT_PARTICLE = """
#version 330
uniform mat4 mvp;
uniform float point_size;
in vec2 in_position;
in vec4 in_color;
out vec4 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 0.0, 1.0);
    gl_PointSize = point_size;
    v_color = in_color;
}
"""

_FRAG_PARTICLE = """
#version 330
in vec4 v_color;
out vec4 frag_color;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d = dot(p, p);
    if (d > 1.0) discard;
    float glow = exp(-d * 2.5);
    frag_color = vec4(v_color.rgb, v_color.a * glow);
}
"""

_VERT_LINE3D = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""


# ── Renderer ─────────────────────────────────────────────

class BellRenderer:
    """Manages all GL state and renders one frame at a time."""

    def __init__(self):
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((W, H), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((W, H)),
        )

        # Particle system for panel 1
        self._max_particles = 200
        self._particles = []  # list of dicts
        self._particle_id = 0

        self._build_programs()
        self._build_geometry()

        # Pre-compute 3D matrices
        aspect = PW / PH
        self.proj_3d = _perspective(35, aspect, 0.1, 50.0)
        self.proj_2d_bell = _ortho(-0.3, 2 * np.pi + 0.3, -1.3, 1.3, -1, 1)
        self.proj_2d_scatter = _ortho(-0.3, 2 * np.pi + 0.3, -1.8, 1.8, -1, 1)
        self.proj_2d_epr = _ortho(-1.2, 1.2, -0.7, 0.7, -1, 1)

    def _build_programs(self):
        self.prog_3d = self.ctx.program(
            vertex_shader=_VERT_3D, fragment_shader=_FRAG_PHONG,
        )
        self.prog_2d = self.ctx.program(
            vertex_shader=_VERT_2D, fragment_shader=_FRAG_FLAT,
        )
        self.prog_point = self.ctx.program(
            vertex_shader=_VERT_POINT, fragment_shader=_FRAG_POINT,
        )
        self.prog_particle = self.ctx.program(
            vertex_shader=_VERT_PARTICLE, fragment_shader=_FRAG_PARTICLE,
        )
        self.prog_line3d = self.ctx.program(
            vertex_shader=_VERT_LINE3D, fragment_shader=_FRAG_FLAT,
        )

    def _build_geometry(self):
        # Sphere
        sv, sn, si = _uv_sphere()
        self._sphere_vbo = self.ctx.buffer(sv.tobytes())
        self._sphere_nbo = self.ctx.buffer(sn.tobytes())
        self._sphere_ibo = self.ctx.buffer(si.tobytes())
        self._sphere_count = len(si)
        self._sphere_vao = self.ctx.vertex_array(
            self.prog_3d,
            [(self._sphere_vbo, "3f", "in_position"),
             (self._sphere_nbo, "3f", "in_normal")],
            index_buffer=self._sphere_ibo,
        )

        # Wireframe circles for sphere decoration
        cv = _circle_outline(64)
        self._circle_vbo = self.ctx.buffer(cv.tobytes())
        self._circle_count = 64

        # Cylinder (arrow shaft)
        cyv, cyn, cyi = _cylinder(0.035, 0.85, 12)
        self._cyl_vbo = self.ctx.buffer(cyv.tobytes())
        self._cyl_nbo = self.ctx.buffer(cyn.tobytes())
        self._cyl_ibo = self.ctx.buffer(cyi.tobytes())
        self._cyl_count = len(cyi)
        self._cyl_vao = self.ctx.vertex_array(
            self.prog_3d,
            [(self._cyl_vbo, "3f", "in_position"),
             (self._cyl_nbo, "3f", "in_normal")],
            index_buffer=self._cyl_ibo,
        )

        # Cone (arrowhead)
        cov, con, coi = _cone(0.08, 0.15, 12)
        self._cone_vbo = self.ctx.buffer(cov.tobytes())
        self._cone_nbo = self.ctx.buffer(con.tobytes())
        self._cone_ibo = self.ctx.buffer(coi.tobytes())
        self._cone_count = len(coi)
        self._cone_vao = self.ctx.vertex_array(
            self.prog_3d,
            [(self._cone_vbo, "3f", "in_position"),
             (self._cone_nbo, "3f", "in_normal")],
            index_buffer=self._cone_ibo,
        )

        # Dynamic buffers for 2D lines, scatter, particles
        self._line2d_vbo = self.ctx.buffer(reserve=4096 * 8)
        self._line2d_vao = self.ctx.vertex_array(
            self.prog_2d, [(self._line2d_vbo, "2f", "in_position")],
        )

        self._scatter_vbo = self.ctx.buffer(reserve=80000 * 20)
        self._scatter_vao = self.ctx.vertex_array(
            self.prog_point,
            [(self._scatter_vbo, "2f 3f", "in_position", "in_color")],
        )

        self._part_vbo = self.ctx.buffer(reserve=self._max_particles * 24)
        self._part_vao = self.ctx.vertex_array(
            self.prog_particle,
            [(self._part_vbo, "2f 4f", "in_position", "in_color")],
        )

    # ── 2D line helper ───────────────────────────────────

    def _draw_lines(self, points, color, mvp, mode=moderngl.LINE_STRIP, line_width=1.5):
        pts = np.asarray(points, dtype="f4").reshape(-1, 2)
        if len(pts) < 2:
            return
        data = pts.tobytes()
        if len(data) > self._line2d_vbo.size:
            self._line2d_vbo.orphan(len(data))
        self._line2d_vbo.write(data)
        self.prog_2d["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog_2d["color"].value = (*color, 1.0)
        self.ctx.line_width = line_width
        self._line2d_vao.render(mode, vertices=len(pts))

    def _draw_dashed_line(self, points, color, mvp, dash_len=0.12):
        """Draw a dashed line by rendering segments."""
        pts = np.asarray(points, dtype="f4").reshape(-1, 2)
        if len(pts) < 2:
            return
        # Build segments with gaps
        segs = []
        on = True
        accum = 0.0
        seg_start = pts[0]
        for i in range(1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[i - 1])
            if on:
                segs.append(pts[i - 1])
                segs.append(pts[i])
            accum += d
            if accum >= dash_len:
                accum = 0.0
                on = not on
        if len(segs) < 2:
            return
        data = np.array(segs, dtype="f4").tobytes()
        if len(data) > self._line2d_vbo.size:
            self._line2d_vbo.orphan(len(data))
        self._line2d_vbo.write(data)
        self.prog_2d["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog_2d["color"].value = (*color, 1.0)
        self.ctx.line_width = 1.5
        self._line2d_vao.render(moderngl.LINES, vertices=len(segs))

    # ── Panel 1: EPR Source ──────────────────────────────

    def _update_particles(self, state):
        frame = state["frame"]
        phase = state["phase"]
        theta_a = state["theta_a"]
        theta_b = state["theta_b"]

        # Spawn new pairs every few frames
        if phase < 4 and frame % 3 == 0:
            self._particle_id += 1
            self._particles.append({
                "id": self._particle_id,
                "born": frame,
                "life": 30,
                "theta_a": theta_a,
                "theta_b": theta_b,
                "result_a": state["last_a"],
                "result_b": state["last_b"],
            })

        # Age and cull
        alive = []
        for p in self._particles:
            age = frame - p["born"]
            if age < p["life"]:
                alive.append(p)
        self._particles = alive[-self._max_particles:]

    def _render_panel1(self, state):
        """EPR source with particle pairs."""
        self.ctx.viewport = (0, PH, PW, PH)
        self.ctx.scissor = (0, PH, PW, PH)

        self._update_particles(state)
        frame = state["frame"]

        mvp = self.proj_2d_epr

        # Source glow (pulsing center)
        pulse = 0.5 + 0.5 * np.sin(frame * 0.15)
        src_pts = _circle_outline(32, 0.06 + 0.02 * pulse)
        pts_2d = src_pts[:, :2]
        # Draw source circle
        self._draw_lines(pts_2d, (1.0, 1.0, 1.0), mvp, mode=moderngl.LINE_LOOP)

        # Detector boxes
        for side, col in [(-1.0, COL_CYAN), (1.0, COL_RED)]:
            box = np.array([
                [side - 0.06, -0.15], [side + 0.06, -0.15],
                [side + 0.06, 0.15], [side - 0.06, 0.15],
                [side - 0.06, -0.15],
            ], dtype="f4")
            self._draw_lines(box, col, mvp, mode=moderngl.LINE_STRIP)

        # Polarizer angle indicators
        theta_a = state["theta_a"]
        theta_b = state["theta_b"]
        for side, angle, col in [(-0.6, theta_a, COL_CYAN), (0.6, theta_b, COL_RED)]:
            dx = 0.1 * np.cos(angle)
            dy = 0.1 * np.sin(angle)
            line = np.array([[side - dx, -dy], [side + dx, dy]], dtype="f4")
            self._draw_lines(line, col, mvp, line_width=2.0)
            # Small circle around polarizer
            pc = _circle_outline(24, 0.12)[:, :2] + np.array([side, 0.0], dtype="f4")
            self._draw_lines(pc, (*col[0:2], col[2] * 0.5) if len(col) == 3 else col, mvp, mode=moderngl.LINE_LOOP)

        # Particles as point sprites
        if self._particles:
            buf = []
            for p in self._particles:
                age = frame - p["born"]
                t = age / p["life"]
                alpha = 1.0 - t * t
                # Photon A goes left, B goes right
                xa = -t * 0.92
                xb = t * 0.92
                y_jitter = 0.02 * np.sin(p["id"] * 7.3 + age * 0.5)
                # A (cyan)
                buf.append((xa, y_jitter, COL_CYAN[0], COL_CYAN[1], COL_CYAN[2], alpha * 0.9))
                # B (red)
                buf.append((xb, y_jitter, COL_RED[0], COL_RED[1], COL_RED[2], alpha * 0.9))
            data = np.array(buf, dtype="f4")
            nbytes = data.nbytes
            if nbytes > self._part_vbo.size:
                self._part_vbo.orphan(nbytes)
            self._part_vbo.write(data.tobytes())
            self.prog_particle["mvp"].write(mvp.T.astype("f4").tobytes())
            self.prog_particle["point_size"].value = 12.0
            self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE
            self._part_vao.render(moderngl.POINTS, vertices=len(buf))
            self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA

    # ── Panel 2: Bloch Spheres ───────────────────────────

    def _render_bloch_sphere(self, center_x, color, meas_angle, outcome, cam_angle):
        """Render one Bloch sphere at given X offset."""
        aspect = PW / PH
        proj = self.proj_3d

        eye = np.array([
            4.0 * np.sin(cam_angle),
            1.5,
            4.0 * np.cos(cam_angle),
        ])
        view = _look_at(eye, [0, 0, 0], [0, 1, 0])

        model_base = _translate(center_x, 0, 0)

        # Sphere wireframe: latitude and longitude circles
        self.ctx.depth_func = '<='
        for i in range(1, 6):
            lat = np.pi * i / 6 - np.pi / 2
            r = np.cos(lat)
            y = np.sin(lat)
            cv = _circle_outline(48, r)
            pts_3d = np.zeros((48, 3), dtype="f4")
            pts_3d[:, 0] = cv[:, 0]
            pts_3d[:, 1] = y
            pts_3d[:, 2] = cv[:, 1]
            # Transform
            mvp = proj @ view @ model_base
            data = pts_3d.tobytes()
            if len(data) > self._line2d_vbo.size:
                self._line2d_vbo.orphan(len(data))
            # Use line3d program
            vbo = self.ctx.buffer(data)
            vao = self.ctx.vertex_array(
                self.prog_line3d, [(vbo, "3f", "in_position")],
            )
            self.prog_line3d["mvp"].write(mvp.T.astype("f4").tobytes())
            self.prog_line3d["color"].value = (*color, 0.15)
            self.ctx.line_width = 1.0
            vao.render(moderngl.LINE_LOOP, vertices=48)
            vbo.release()
            vao.release()

        # Longitude circles
        for j in range(6):
            phi = np.pi * j / 6
            cv = _circle_outline(48, 1.0)
            pts_3d = np.zeros((48, 3), dtype="f4")
            pts_3d[:, 0] = cv[:, 0] * np.cos(phi)
            pts_3d[:, 1] = cv[:, 1]
            pts_3d[:, 2] = cv[:, 0] * np.sin(phi)
            mvp = proj @ view @ model_base
            vbo = self.ctx.buffer(pts_3d.tobytes())
            vao = self.ctx.vertex_array(
                self.prog_line3d, [(vbo, "3f", "in_position")],
            )
            self.prog_line3d["mvp"].write(mvp.T.astype("f4").tobytes())
            self.prog_line3d["color"].value = (*color, 0.15)
            vao.render(moderngl.LINE_LOOP, vertices=48)
            vbo.release()
            vao.release()

        # State arrow: measurement axis rotated by meas_angle in XZ plane
        # outcome +1 → up along axis, -1 → down
        arrow_dir = outcome  # +1 or -1
        arrow_rot = _rot_z(meas_angle) if meas_angle != 0 else np.eye(4, dtype="f4")
        arrow_model = model_base @ arrow_rot

        # Arrow shaft
        shaft_model = arrow_model.copy()
        if arrow_dir < 0:
            shaft_model = shaft_model @ _scale(1, -1, 1)
        mvp = proj @ view @ shaft_model
        self.prog_3d["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog_3d["model"].write(shaft_model.T.astype("f4").tobytes())
        self.prog_3d["color"].value = color
        self.prog_3d["alpha"].value = 1.0
        self.prog_3d["light_dir"].value = (0.5, 1.0, 0.8)
        self._cyl_vao.render()

        # Arrowhead
        head_model = shaft_model @ _translate(0, 0.85, 0)
        mvp = proj @ view @ head_model
        self.prog_3d["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog_3d["model"].write(head_model.T.astype("f4").tobytes())
        self._cone_vao.render()

        self.ctx.depth_func = '<'

    def _render_panel2(self, state):
        """Two Bloch spheres side by side."""
        self.ctx.viewport = (PW, PH, PW, PH)
        self.ctx.scissor = (PW, PH, PW, PH)

        frame = state["frame"]
        cam_angle = np.radians(5.0 * frame / 1500.0) + 0.3

        self._render_bloch_sphere(
            -1.4, COL_CYAN, state["bloch_a"], state["last_a"], cam_angle,
        )
        self._render_bloch_sphere(
            1.4, COL_RED, state["bloch_b"], state["last_b"], cam_angle,
        )

    # ── Panel 3: Scatter Plot ────────────────────────────

    def _render_panel3(self, state):
        """Scatter plot of measurement outcomes."""
        self.ctx.viewport = (0, 0, PW, PH)
        self.ctx.scissor = (0, 0, PW, PH)

        mvp = self.proj_2d_scatter

        # Axes
        x_axis = np.array([[0, 0], [2 * np.pi, 0]], dtype="f4")
        y_axis = np.array([[0, -1.5], [0, 1.5]], dtype="f4")
        self._draw_lines(x_axis, COL_GREY, mvp, line_width=1.0)
        self._draw_lines(y_axis, COL_GREY, mvp, line_width=1.0)

        # Concordance lines at ±1
        self._draw_lines(
            np.array([[0, 1], [2 * np.pi, 1]], dtype="f4"),
            (0.25, 0.25, 0.25), mvp, line_width=1.0,
        )
        self._draw_lines(
            np.array([[0, -1], [2 * np.pi, -1]], dtype="f4"),
            (0.25, 0.25, 0.25), mvp, line_width=1.0,
        )

        # Scatter points
        thetas = state["scatter_thetas"]
        ab = state["scatter_ab"]
        if len(thetas) == 0:
            return

        # Limit displayed points (performance)
        max_pts = 15000
        if len(thetas) > max_pts:
            idx = np.linspace(0, len(thetas) - 1, max_pts, dtype=int)
            thetas = thetas[idx]
            ab = ab[idx]

        rng = np.random.default_rng(123)
        jitter = rng.uniform(-0.35, 0.35, size=len(ab))
        y_vals = ab.astype("f4") + jitter.astype("f4")

        # Colors: green if concordant (ab=+1), red if discordant (ab=-1)
        colors = np.zeros((len(ab), 3), dtype="f4")
        concordant = ab > 0
        colors[concordant] = COL_GREEN
        colors[~concordant] = COL_RED

        data = np.column_stack([thetas.astype("f4"), y_vals, colors])
        buf = data.astype("f4").tobytes()
        if len(buf) > self._scatter_vbo.size:
            self._scatter_vbo.orphan(len(buf))
        self._scatter_vbo.write(buf)
        self.prog_point["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog_point["point_size"].value = 3.0
        self._scatter_vao.render(moderngl.POINTS, vertices=len(ab))

    # ── Panel 4: Bell Curve ──────────────────────────────

    def _render_panel4(self, state):
        """E(θ) correlation curves + CHSH indicator."""
        self.ctx.viewport = (PW, 0, PW, PH)
        self.ctx.scissor = (PW, 0, PW, PH)

        mvp = self.proj_2d_bell

        # Axes
        x_axis = np.array([[0, 0], [2 * np.pi, 0]], dtype="f4")
        y_axis = np.array([[0, -1.2], [0, 1.2]], dtype="f4")
        self._draw_lines(x_axis, COL_GREY, mvp, line_width=1.0)
        self._draw_lines(y_axis, COL_GREY, mvp, line_width=1.0)

        # Classical bound shading (lines at ±1)
        bound_top = np.array([[0, 1], [2 * np.pi, 1]], dtype="f4")
        bound_bot = np.array([[0, -1], [2 * np.pi, -1]], dtype="f4")
        self._draw_lines(bound_top, (0.25, 0.25, 0.25), mvp, line_width=1.0)
        self._draw_lines(bound_bot, (0.25, 0.25, 0.25), mvp, line_width=1.0)

        # Theoretical curves
        t = np.linspace(0, 2 * np.pi, 300, dtype="f4")

        # Classical: E = 1 - 2|θ|/π (folded)
        from bell import classical_correlation, quantum_correlation
        e_cl = classical_correlation(t).astype("f4")
        cl_pts = np.column_stack([t, e_cl])
        self._draw_dashed_line(cl_pts, COL_GREY, mvp, dash_len=0.08)

        # Quantum: E = cos(θ)
        e_qm = quantum_correlation(t).astype("f4")
        qm_pts = np.column_stack([t, e_qm])
        self._draw_lines(qm_pts, COL_CYAN, mvp, line_width=2.0)

        # Binned data points
        centers = state["bin_centers"]
        E_binned = state["E_binned"]
        mask = ~np.isnan(E_binned)
        if np.any(mask):
            n_show = mask.sum()
            pts = np.column_stack([
                centers[mask].astype("f4"),
                E_binned[mask].astype("f4"),
            ])
            colors = np.full((n_show, 3), COL_YELLOW, dtype="f4")
            data = np.column_stack([pts, colors]).astype("f4")
            buf = data.tobytes()
            if len(buf) > self._scatter_vbo.size:
                self._scatter_vbo.orphan(len(buf))
            self._scatter_vbo.write(buf)
            self.prog_point["mvp"].write(mvp.T.astype("f4").tobytes())
            self.prog_point["point_size"].value = 4.0
            self._scatter_vao.render(moderngl.POINTS, vertices=n_show)

    # ── Panel borders ────────────────────────────────────

    def _render_borders(self):
        """Draw thin lines separating the 4 panels."""
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.scissor = (0, 0, W, H)
        mvp = _ortho(0, W, 0, H, -1, 1)
        # Horizontal
        self._draw_lines(
            np.array([[0, PH], [W, PH]], dtype="f4"),
            (0.2, 0.2, 0.2), mvp, line_width=1.5,
        )
        # Vertical
        self._draw_lines(
            np.array([[PW, 0], [PW, H]], dtype="f4"),
            (0.2, 0.2, 0.2), mvp, line_width=1.5,
        )

    # ── Text overlay (PIL) ───────────────────────────────

    def _overlay_text(self, img, state):
        """Add labels and values using PIL."""
        draw = ImageDraw.Draw(img)

        def txt(x, y, s, font=_F18, color=(255, 255, 255)):
            draw.text((x + 1, y + 1), s, fill=(0, 0, 0), font=font)
            draw.text((x, y), s, fill=color, font=font)

        frame = state["frame"]
        phase = state["phase"]
        theta = state["theta"]
        n_meas = state["n_measurements"]
        chsh_s = state["chsh_s"]

        # Panel 1 title
        txt(15, 8, "Fonte EPR", _F22)
        txt(15, 35, f"\u03b8_a={np.degrees(state['theta_a']):.1f}\u00b0  "
                     f"\u03b8_b={np.degrees(state['theta_b']):.1f}\u00b0", _F14)
        txt(15, 52, f"N={n_meas}", _F14, (180, 180, 180))

        # Panel 2 title
        txt(PW + 15, 8, "Esferas de Bloch", _F22)
        txt(PW + 15, 35, "A (cyan)         B (vermelho)", _F14)

        # Panel 3 title
        txt(15, PH + 8, "Detec\u00e7\u00f5es (a\u00d7b)", _F22)
        txt(15, PH + 35, "\u03b8 \u2192", _F14, (180, 180, 180))

        # Panel 4 title
        txt(PW + 15, PH + 8, "Correla\u00e7\u00e3o E(\u03b8)", _F22)

        # Classical vs quantum legend
        txt(PW + 15, PH + 35, "--- cl\u00e1ssico", _F14, (102, 102, 102))
        txt(PW + 175, PH + 35, "\u2014 qu\u00e2ntico", _F14, (0, 229, 255))

        # CHSH S value
        if phase >= 3:
            s_color = (255, 215, 64) if abs(chsh_s) > 2.0 else (255, 255, 255)
            txt(PW + PW - 220, PH + PH - 55, f"S = {chsh_s:.2f}", _F22, s_color)
            if abs(chsh_s) > 2.0:
                txt(PW + PW - 220, PH + PH - 30, "S > 2: viola\u00e7\u00e3o!", _F14, (255, 215, 64))

        # Phase indicator
        phase_names = {1: "Fase 1: \u03b8=0", 2: "Fase 2: varredura \u03b8",
                       3: "Fase 3: CHSH", 4: "Resultado final"}
        txt(W - 280, 8, phase_names.get(phase, ""), _F14, (120, 120, 120))

    # ── Main render ──────────────────────────────────────

    def render_frame(self, state):
        """Render one frame → PIL Image (1920×1080)."""
        self.fbo.use()
        self.ctx.clear(*BG)

        # Render panels (order: back-to-front within each panel)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._render_panel4(state)
        self._render_panel3(state)
        self._render_panel1(state)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._render_panel2(state)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._render_borders()

        # Read FBO
        raw = self.fbo.read(components=3)
        img = Image.frombytes("RGB", (W, H), raw)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Text overlay
        self._overlay_text(img, state)
        return img

    def release(self):
        self.fbo.release()
        self.ctx.release()
