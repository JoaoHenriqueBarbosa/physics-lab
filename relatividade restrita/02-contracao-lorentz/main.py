"""
02 · Contração de Lorentz — Relatividade Restrita

Um cubo e uma esfera lado a lado, com wireframes-fantasma mostrando
o tamanho de repouso (L₀) e as formas sólidas mostrando o comprimento
contraído L = L₀ / γ.

A contração acontece inteiramente no vertex shader da GPU:
cada vértice é comprimido ao longo do eixo de movimento pelo fator
1/γ = √(1 − v²/c²).

Controles:
    ←/→        Ajustar velocidade (v/c)
    A          Auto-rotação on/off
    SPACE      Pausar / continuar
    R          Resetar
    B          Bloom on/off
    Mouse      Arrastar para orbitar câmera
    Scroll     Zoom
    ESC        Sair
"""

import math
import sys
from pathlib import Path

import numpy as np
import pygame
import moderngl

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
W, H = 1600, 900
BLOOM_DS = 4
SHADER_DIR = Path(__file__).parent / "shaders"

COLOR_CYAN = (0.0, 0.75, 1.0)
COLOR_AMBER = (1.0, 0.68, 0.0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matrix helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mat4_perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m

def mat4_look_at(eye, target, up):
    eye, target, up = (np.asarray(v, dtype="f4") for v in (eye, target, up))
    f = target - eye;  f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype="f4")
    m[0, :3] = s;   m[0, 3] = -s.dot(eye)
    m[1, :3] = u;   m[1, 3] = -u.dot(eye)
    m[2, :3] = -f;  m[2, 3] = f.dot(eye)
    return m

def mat4_translate(tx, ty, tz):
    m = np.eye(4, dtype="f4"); m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m

def mat4_scale(sx, sy, sz):
    m = np.eye(4, dtype="f4"); m[0, 0] = sx; m[1, 1] = sy; m[2, 2] = sz
    return m

def mat4_rotate_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    m = np.eye(4, dtype="f4")
    m[0, 0] = c;  m[0, 2] = s
    m[2, 0] = -s; m[2, 2] = c
    return m

def _gl(m):
    return np.ascontiguousarray(m.T).tobytes()

def _load(name):
    return (SHADER_DIR / name).read_text()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mesh generators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cube_faces():
    """36 vertices: position(3) + normal(3)."""
    faces = [
        ([[-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]], [ 0, 0, 1]),
        ([[ 1,-1,-1],[-1,-1,-1],[-1, 1,-1],[ 1, 1,-1]], [ 0, 0,-1]),
        ([[ 1,-1, 1],[ 1,-1,-1],[ 1, 1,-1],[ 1, 1, 1]], [ 1, 0, 0]),
        ([[-1,-1,-1],[-1,-1, 1],[-1, 1, 1],[-1, 1,-1]], [-1, 0, 0]),
        ([[-1, 1, 1],[ 1, 1, 1],[ 1, 1,-1],[-1, 1,-1]], [ 0, 1, 0]),
        ([[-1,-1,-1],[ 1,-1,-1],[ 1,-1, 1],[-1,-1, 1]], [ 0,-1, 0]),
    ]
    verts = []
    for quad, n in faces:
        for idx in (0, 1, 2, 0, 2, 3):
            verts.extend(quad[idx] + n)
    return np.array(verts, dtype="f4")

def _cube_edges():
    """24 vertices: position(3) for 12 edges."""
    v = [
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    verts = []
    for a, b in edges:
        verts.extend(v[a]); verts.extend(v[b])
    return np.array(verts, dtype="f4")

def _sphere_faces(rings=32, sectors=48):
    """Indexed mesh: vertices position(3)+normal(3), indices."""
    verts = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            verts.extend([x, y, z, x, y, z])  # pos = normal for unit sphere
    indices = []
    for r in range(rings):
        for s in range(sectors):
            i = r * (sectors + 1) + s
            indices.extend([i, i + sectors + 1, i + 1,
                            i + 1, i + sectors + 1, i + sectors + 2])
    return np.array(verts, dtype="f4"), np.array(indices, dtype="i4")

def _sphere_wireframe(lat_step=30, lon_step=30, segs=48):
    """Latitude + longitude lines: position(3) pairs for GL_LINES."""
    verts = []
    # latitude circles
    for lat_deg in range(-90 + lat_step, 90, lat_step):
        lat = math.radians(lat_deg)
        y = math.sin(lat); rc = math.cos(lat)
        for i in range(segs):
            a1 = 2.0 * math.pi * i / segs
            a2 = 2.0 * math.pi * (i + 1) / segs
            verts.extend([rc * math.cos(a1), y, rc * math.sin(a1)])
            verts.extend([rc * math.cos(a2), y, rc * math.sin(a2)])
    # longitude half-circles
    for lon_deg in range(0, 360, lon_step):
        lon = math.radians(lon_deg)
        for i in range(segs):
            p1 = math.pi * (-0.5 + i / segs)
            p2 = math.pi * (-0.5 + (i + 1) / segs)
            verts.extend([math.cos(p1)*math.cos(lon), math.sin(p1), math.cos(p1)*math.sin(lon)])
            verts.extend([math.cos(p2)*math.cos(lon), math.sin(p2), math.cos(p2)*math.sin(lon)])
    return np.array(verts, dtype="f4")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class App:
    def __init__(self):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        self.surface = pygame.display.set_mode(
            (W, H), pygame.DOUBLEBUF | pygame.OPENGL
        )
        pygame.display.set_caption("02 · Contração de Lorentz — Relatividade Restrita")

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._build_programs()
        self._build_geometry()
        self._build_fbos()
        self._build_hud()

        # physics
        self.speed = 0.0
        self.paused = False
        self.bloom_on = True
        self.auto_rotate = True
        self.rotation = 0.0
        self.wall_time = 0.0

        # camera
        self.cam_theta = 0.35
        self.cam_phi = 0.38
        self.cam_dist = 18.0
        self.dragging = False
        self.last_mouse = (0, 0)

        self.clock = pygame.time.Clock()

    # ── programs ─────────────────────────────────────────────
    def _build_programs(self):
        screen_v = _load("screen.vert")
        self.grid_prog  = self.ctx.program(vertex_shader=_load("scene.vert"),
                                           fragment_shader=_load("grid.frag"))
        self.cube_prog  = self.ctx.program(vertex_shader=_load("cube.vert"),
                                           fragment_shader=_load("cube.frag"))
        self.wire_prog  = self.ctx.program(vertex_shader=_load("wire.vert"),
                                           fragment_shader=_load("wire.frag"))
        self.stars_prog = self.ctx.program(vertex_shader=screen_v,
                                           fragment_shader=_load("stars.frag"))
        self.bright_prog = self.ctx.program(vertex_shader=screen_v,
                                            fragment_shader=_load("bloom_extract.frag"))
        self.blur_prog   = self.ctx.program(vertex_shader=screen_v,
                                            fragment_shader=_load("bloom_blur.frag"))
        self.comp_prog   = self.ctx.program(vertex_shader=screen_v,
                                            fragment_shader=_load("composite.frag"))

    # ── geometry ─────────────────────────────────────────────
    def _build_geometry(self):
        # full-screen quad
        fsq = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype="f4")
        self.fsq_buf = self.ctx.buffer(fsq)
        def _sq(prog):
            return self.ctx.vertex_array(prog,
                [(self.fsq_buf, "2f 2f", "in_position", "in_texcoord")])
        self.stars_vao = _sq(self.stars_prog)
        self.bright_vao = _sq(self.bright_prog)
        self.blur_vao = _sq(self.blur_prog)
        self.comp_vao = _sq(self.comp_prog)

        # grid
        S = 60.0
        grid = np.array([-S,0,-S, S,0,-S, -S,0,S, S,0,S], dtype="f4")
        grid_buf = self.ctx.buffer(grid)
        self.grid_vao = self.ctx.vertex_array(
            self.grid_prog, [(grid_buf, "3f", "in_position")])

        # cube faces
        cf = _cube_faces()
        self.cube_vao = self.ctx.vertex_array(
            self.cube_prog, [(self.ctx.buffer(cf), "3f 3f", "in_position", "in_normal")])

        # cube wireframe
        ce = _cube_edges()
        self.cube_wire_vao = self.ctx.vertex_array(
            self.wire_prog, [(self.ctx.buffer(ce), "3f", "in_position")])

        # sphere faces (indexed)
        sv, si = _sphere_faces()
        self.sphere_vao = self.ctx.vertex_array(
            self.cube_prog,
            [(self.ctx.buffer(sv), "3f 3f", "in_position", "in_normal")],
            index_buffer=self.ctx.buffer(si),
        )

        # sphere wireframe
        sw = _sphere_wireframe()
        self.sphere_wire_vao = self.ctx.vertex_array(
            self.wire_prog, [(self.ctx.buffer(sw), "3f", "in_position")])

    # ── FBOs ─────────────────────────────────────────────────
    def _build_fbos(self):
        self.scene_tex = self.ctx.texture((W, H), 4, dtype="f2")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_fbo = self.ctx.framebuffer(
            self.scene_tex, self.ctx.depth_renderbuffer((W, H)))

        bw, bh = W // BLOOM_DS, H // BLOOM_DS
        self.bloom_a_tex = self.ctx.texture((bw, bh), 4, dtype="f2")
        self.bloom_b_tex = self.ctx.texture((bw, bh), 4, dtype="f2")
        for t in (self.bloom_a_tex, self.bloom_b_tex):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bloom_a_fbo = self.ctx.framebuffer(self.bloom_a_tex)
        self.bloom_b_fbo = self.ctx.framebuffer(self.bloom_b_tex)

    # ── HUD ──────────────────────────────────────────────────
    def _build_hud(self):
        self.font_title = pygame.font.SysFont("monospace", 30, bold=True)
        self.font_med   = pygame.font.SysFont("monospace", 22)
        self.font_sm    = pygame.font.SysFont("monospace", 18)
        self.hud_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        self.hud_tex  = self.ctx.texture((W, H), 4)
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ── helpers ──────────────────────────────────────────────
    @property
    def gamma(self):
        v2 = self.speed * self.speed
        return 1.0 / math.sqrt(1.0 - v2) if v2 < 1.0 else float("inf")

    @property
    def contraction(self):
        v2 = self.speed * self.speed
        return math.sqrt(1.0 - v2) if v2 < 1.0 else 0.001

    def _cam_eye(self):
        x = self.cam_dist * math.sin(self.cam_theta) * math.cos(self.cam_phi)
        y = self.cam_dist * math.sin(self.cam_phi)
        z = self.cam_dist * math.cos(self.cam_theta) * math.cos(self.cam_phi)
        return np.array([x, y, z], dtype="f4")

    def _project(self, point, vp):
        p = vp @ np.array([*point, 1.0], dtype="f4")
        if abs(p[3]) > 1e-6: p /= p[3]
        return int((p[0] + 1.0) * W / 2.0), int((1.0 - p[1]) * H / 2.0)

    # ── input ────────────────────────────────────────────────
    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return False
                elif ev.key == pygame.K_SPACE: self.paused = not self.paused
                elif ev.key == pygame.K_r:
                    self.speed = 0.0; self.rotation = 0.0
                elif ev.key == pygame.K_b: self.bloom_on = not self.bloom_on
                elif ev.key == pygame.K_a: self.auto_rotate = not self.auto_rotate
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1: self.dragging = True; self.last_mouse = ev.pos
                elif ev.button == 4: self.cam_dist = max(8, self.cam_dist - 1)
                elif ev.button == 5: self.cam_dist = min(40, self.cam_dist + 1)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1: self.dragging = False
            elif ev.type == pygame.MOUSEMOTION and self.dragging:
                dx = ev.pos[0] - self.last_mouse[0]
                dy = ev.pos[1] - self.last_mouse[1]
                self.cam_theta -= dx * 0.005
                self.cam_phi = max(0.05, min(1.4, self.cam_phi + dy * 0.005))
                self.last_mouse = ev.pos
        keys = pygame.key.get_pressed()
        step = 0.002
        if keys[pygame.K_RIGHT] or keys[pygame.K_UP]:
            self.speed = min(0.995, self.speed + step)
        if keys[pygame.K_LEFT] or keys[pygame.K_DOWN]:
            self.speed = max(0.0, self.speed - step)
        return True

    # ── update ───────────────────────────────────────────────
    def _update(self, dt):
        self.wall_time += dt
        if not self.paused and self.auto_rotate:
            self.rotation += dt * 0.4

    # ── render helpers ───────────────────────────────────────
    def _set_wire(self, model, vp, contraction, motion, color, alpha):
        p = self.wire_prog
        p["u_model"].write(_gl(model))
        p["u_viewproj"].write(_gl(vp))
        p["u_contraction"].value = contraction
        p["u_motion_dir"].value = motion
        p["u_color"].value = color
        p["u_alpha"].value = alpha

    def _set_cube(self, model, vp, contraction, motion, color, alpha, cam):
        p = self.cube_prog
        p["u_model"].write(_gl(model))
        p["u_viewproj"].write(_gl(vp))
        p["u_contraction"].value = contraction
        p["u_motion_dir"].value = motion
        p["u_color"].value = color
        p["u_alpha"].value = alpha
        p["u_cam_pos"].value = tuple(cam)

    # ── render ───────────────────────────────────────────────
    def _render(self):
        eye = self._cam_eye()
        view = mat4_look_at(eye, [0, 1.8, 0], [0, 1, 0])
        proj = mat4_perspective(45, W / H, 0.1, 200)
        vp = proj @ view
        c = self.contraction
        motion = (1.0, 0.0, 0.0)
        obj_scale = 1.8

        model_cube   = mat4_translate(-4, 2.2, 0) @ mat4_rotate_y(self.rotation) @ mat4_scale(obj_scale, obj_scale, obj_scale)
        model_sphere = mat4_translate( 4, 2.2, 0) @ mat4_rotate_y(self.rotation) @ mat4_scale(obj_scale, obj_scale, obj_scale)

        # ── scene FBO ────────────────────────────────────────
        self.scene_fbo.use()
        self.ctx.clear(0, 0, 0, 1)

        # stars
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.stars_prog["u_time"].value = self.wall_time
        self.stars_vao.render(moderngl.TRIANGLE_STRIP)

        # grid
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.grid_prog["u_model"].write(_gl(np.eye(4, dtype="f4")))
        self.grid_prog["u_viewproj"].write(_gl(vp))
        self.grid_vao.render(moderngl.TRIANGLE_STRIP)

        # ── ghost wireframes (rest frame, contraction=1) ─────
        self.ctx.depth_mask = False
        ghost_cyan  = (0.0, 0.25, 0.4)
        ghost_amber = (0.4, 0.28, 0.0)

        self._set_wire(model_cube, vp, 1.0, motion, ghost_cyan, 0.25)
        self.cube_wire_vao.render(moderngl.LINES)

        self._set_wire(model_sphere, vp, 1.0, motion, ghost_amber, 0.20)
        self.sphere_wire_vao.render(moderngl.LINES)

        # ── contracted solid shapes ──────────────────────────
        self._set_cube(model_cube, vp, c, motion, COLOR_CYAN, 0.28, eye)
        self.ctx.cull_face = "back"
        self.ctx.enable(moderngl.CULL_FACE)
        self.cube_vao.render(moderngl.TRIANGLES)

        self._set_cube(model_sphere, vp, c, motion, COLOR_AMBER, 0.28, eye)
        self.sphere_vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.CULL_FACE)

        # ── contracted wireframes (bright, on top) ───────────
        self._set_wire(model_cube, vp, c, motion, (0.0, 0.9, 1.0), 0.85)
        self.cube_wire_vao.render(moderngl.LINES)

        self._set_wire(model_sphere, vp, c, motion, (1.0, 0.72, 0.0), 0.65)
        self.sphere_wire_vao.render(moderngl.LINES)

        self.ctx.depth_mask = True

        # ── bloom ────────────────────────────────────────────
        self.ctx.disable(moderngl.DEPTH_TEST)
        if self.bloom_on:
            bw, bh = W // BLOOM_DS, H // BLOOM_DS

            self.bloom_a_fbo.use()
            self.scene_tex.use(0)
            self.bright_prog["u_texture"].value = 0
            self.bright_prog["u_threshold"].value = 0.40
            self.bright_vao.render(moderngl.TRIANGLE_STRIP)

            # 2-pass blur × 2 iterations
            for spread in (1.0, 2.0):
                self.bloom_b_fbo.use()
                self.bloom_a_tex.use(0)
                self.blur_prog["u_texture"].value = 0
                self.blur_prog["u_direction"].value = (spread / bw, 0.0)
                self.blur_vao.render(moderngl.TRIANGLE_STRIP)

                self.bloom_a_fbo.use()
                self.bloom_b_tex.use(0)
                self.blur_prog["u_texture"].value = 0
                self.blur_prog["u_direction"].value = (0.0, spread / bh)
                self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        # ── HUD ──────────────────────────────────────────────
        self._update_hud(vp)

        # ── composite ────────────────────────────────────────
        self.ctx.screen.use()
        self.ctx.clear(0, 0, 0, 1)
        self.scene_tex.use(0)
        self.bloom_a_tex.use(1)
        self.hud_tex.use(2)
        cp = self.comp_prog
        cp["u_scene"].value = 0
        cp["u_bloom"].value = 1
        cp["u_hud"].value = 2
        cp["u_bloom_strength"].value = 1.3
        cp["u_bloom_on"].value = int(self.bloom_on)
        self.comp_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()

    # ── HUD ──────────────────────────────────────────────────
    def _update_hud(self, vp):
        s = self.hud_surf
        s.fill((0, 0, 0, 0))
        white = (230, 230, 240)
        dim   = (120, 120, 150)
        cyan  = (0, 200, 255)
        amber = (255, 180, 0)

        s.blit(self.font_title.render("CONTRAÇÃO DE LORENTZ", True, white), (20, 12))
        s.blit(self.font_sm.render("Relatividade Restrita", True, dim), (20, 46))

        for i, ln in enumerate([
            "←/→   Velocidade", "A      Auto-rot",
            "SPACE  Pausar", "R      Reset",
            "B      Bloom", "Mouse  Orbitar",
        ]):
            s.blit(self.font_sm.render(ln, True, dim), (W - 240, 12 + i * 22))

        # labels above objects
        sx1, sy1 = self._project((-4, 4.5, 0), vp)
        sx2, sy2 = self._project(( 4, 4.5, 0), vp)
        lbl1 = self.font_med.render("Cubo", True, cyan)
        lbl2 = self.font_med.render("Esfera", True, amber)
        s.blit(lbl1, (sx1 - lbl1.get_width() // 2, sy1))
        s.blit(lbl2, (sx2 - lbl2.get_width() // 2, sy2))

        # dimensions below objects
        L0 = 1.8 * 2  # obj_scale * 2 (cube side = 2*scale)
        L = L0 * self.contraction
        st1, st1y = self._project((-4, -0.2, 0), vp)
        st2, st2y = self._project(( 4, -0.2, 0), vp)
        t1 = self.font_sm.render(f"L = {L:.2f}", True, cyan)
        t2 = self.font_sm.render(f"L = {L:.2f}", True, amber)
        s.blit(t1, (st1 - t1.get_width() // 2, st1y))
        s.blit(t2, (st2 - t2.get_width() // 2, st2y))

        # bottom bar
        g = self.gamma
        ratio = self.contraction
        info = f"v/c = {self.speed:.4f}    γ = {g:.4f}    L₀ = {L0:.2f}    L = {L:.2f}    L/L₀ = {ratio:.4f}"
        t = self.font_med.render(info, True, white)
        s.blit(t, (W // 2 - t.get_width() // 2, H - 45))

        # speed bar
        bar_w, bar_x, bar_y = 300, W // 2 - 150, H - 65
        pygame.draw.rect(s, (40, 40, 60), (bar_x, bar_y, bar_w, 8), border_radius=4)
        fill = int(bar_w * self.speed / 0.995)
        if fill > 0:
            c_bar = (int(255 * self.speed), int(200 * (1 - self.speed)), 50)
            pygame.draw.rect(s, c_bar, (bar_x, bar_y, fill, 8), border_radius=4)

        if self.paused:
            pt = self.font_title.render("PAUSADO", True, (255, 90, 90))
            s.blit(pt, (W // 2 - pt.get_width() // 2, 12))

        data = pygame.image.tobytes(s, "RGBA", True)
        self.hud_tex.write(data)

    # ── main loop ────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            running = self._handle_events()
            self._update(dt)
            self._render()
        pygame.quit()


if __name__ == "__main__":
    App().run()
