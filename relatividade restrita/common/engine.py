"""
Shared engine for the Relatividade Restrita experiment series.

Provides:
    BaseApp   – window, OpenGL context, bloom pipeline, camera orbit,
                HUD infrastructure, main loop.
    Helpers   – matrix builders, GL upload, shader loaders.
    Builders  – build_grid(), build_clock() for reusable scene elements.

Subclass BaseApp and override:
    setup()              – create shaders, geometry, state
    on_key(key)          – custom key handling
    update(dt)           – physics / animation
    render_scene(vp, eye)– draw into the scene FBO
    render_hud(surf, vp) – draw text on the HUD surface
"""

import math
from pathlib import Path

import numpy as np
import pygame
import moderngl

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
W, H = 1600, 900
BLOOM_DS = 4
_COMMON_SHADERS = Path(__file__).parent / "shaders"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matrix helpers  (row‑major; call gl_bytes() before uploading)
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
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype="f4")
    m[0, :3] = s;  m[0, 3] = -s.dot(eye)
    m[1, :3] = u;  m[1, 3] = -u.dot(eye)
    m[2, :3] = -f; m[2, 3] = f.dot(eye)
    return m

def mat4_translate(tx, ty, tz):
    m = np.eye(4, dtype="f4"); m[0,3]=tx; m[1,3]=ty; m[2,3]=tz; return m

def mat4_scale(sx, sy, sz):
    m = np.eye(4, dtype="f4"); m[0,0]=sx; m[1,1]=sy; m[2,2]=sz; return m

def mat4_rotate_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    m = np.eye(4, dtype="f4")
    m[0,0]=c; m[0,2]=s; m[2,0]=-s; m[2,2]=c
    return m

def gl_bytes(m):
    """Row‑major numpy → column‑major bytes for GL uniforms."""
    return np.ascontiguousarray(m.T).tobytes()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shader loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_common(name):
    """Load a shader from common/shaders/."""
    return (_COMMON_SHADERS / name).read_text()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BaseApp
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseApp:
    TITLE      = "Experimento"
    SHADER_DIR = None          # override → Path to experiment shaders/

    CAM_TARGET = (0.0, 2.0, 0.0)
    CAM_THETA  = 0.0
    CAM_PHI    = 0.35
    CAM_DIST   = 18.0
    CAM_PHI_RANGE  = (0.05, 1.4)
    CAM_DIST_RANGE = (5, 40)
    SPEED_RANGE = (0.0, 0.995)

    # ── init ─────────────────────────────────────────────────
    def __init__(self):
        pygame.init()
        for attr, val in [
            (pygame.GL_CONTEXT_MAJOR_VERSION, 3),
            (pygame.GL_CONTEXT_MINOR_VERSION, 3),
            (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE),
        ]:
            pygame.display.gl_set_attribute(attr, val)
        self.surface = pygame.display.set_mode(
            (W, H), pygame.DOUBLEBUF | pygame.OPENGL
        )
        pygame.display.set_caption(self.TITLE)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._init_pipeline()

        # camera
        self.cam_theta  = float(self.CAM_THETA)
        self.cam_phi    = float(self.CAM_PHI)
        self.cam_dist   = float(self.CAM_DIST)
        self.cam_target = np.array(self.CAM_TARGET, dtype="f4")
        self._drag = False
        self._last_m = (0, 0)

        # common state
        self.speed     = 0.0
        self.paused    = False
        self.bloom_on  = True
        self.wall_time = 0.0

        # fonts
        self.font_title = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_med   = pygame.font.SysFont("monospace", 20)
        self.font_sm    = pygame.font.SysFont("monospace", 16)
        self.hud_surf   = pygame.Surface((W, H), pygame.SRCALPHA)
        self.hud_tex    = self.ctx.texture((W, H), 4)
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.pg_clock = pygame.time.Clock()
        self.setup()

    # ── internal pipeline ────────────────────────────────────
    def _init_pipeline(self):
        sv = load_common("screen.vert")
        self.stars_prog  = self.ctx.program(vertex_shader=sv, fragment_shader=load_common("stars.frag"))
        self.bright_prog = self.ctx.program(vertex_shader=sv, fragment_shader=load_common("bloom_extract.frag"))
        self.blur_prog   = self.ctx.program(vertex_shader=sv, fragment_shader=load_common("bloom_blur.frag"))
        self.comp_prog   = self.ctx.program(vertex_shader=sv, fragment_shader=load_common("composite.frag"))

        fsq = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype="f4")
        buf = self.ctx.buffer(fsq)
        def _sq(p):
            return self.ctx.vertex_array(p, [(buf, "2f 2f", "in_position", "in_texcoord")])
        self.stars_vao  = _sq(self.stars_prog)
        self.bright_vao = _sq(self.bright_prog)
        self.blur_vao   = _sq(self.blur_prog)
        self.comp_vao   = _sq(self.comp_prog)

        # scene FBO (HDR)
        self.scene_tex = self.ctx.texture((W, H), 4, dtype="f2")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_fbo = self.ctx.framebuffer(
            self.scene_tex, self.ctx.depth_renderbuffer((W, H)))

        # bloom FBOs
        bw, bh = W // BLOOM_DS, H // BLOOM_DS
        self.bloom_a_tex = self.ctx.texture((bw, bh), 4, dtype="f2")
        self.bloom_b_tex = self.ctx.texture((bw, bh), 4, dtype="f2")
        for t in (self.bloom_a_tex, self.bloom_b_tex):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bloom_a_fbo = self.ctx.framebuffer(self.bloom_a_tex)
        self.bloom_b_fbo = self.ctx.framebuffer(self.bloom_b_tex)

    # ── reusable builders ────────────────────────────────────
    def build_grid(self):
        """Return (program, vao) for an anti‑aliased infinite grid."""
        prog = self.ctx.program(
            vertex_shader=load_common("grid.vert"),
            fragment_shader=load_common("grid.frag"))
        S = 60.0
        buf = self.ctx.buffer(
            np.array([-S,0,-S, S,0,-S, -S,0,S, S,0,S], dtype="f4"))
        vao = self.ctx.vertex_array(prog, [(buf, "3f", "in_position")])
        return prog, vao

    def build_clock(self):
        """Return (program, vao) for a procedural analog clock quad."""
        prog = self.ctx.program(
            vertex_shader=load_common("world.vert"),
            fragment_shader=load_common("clock.frag"))
        buf = self.ctx.buffer(np.array(
            [-1,-1,0,0,0, 1,-1,0,1,0, -1,1,0,0,1, 1,1,0,1,1], dtype="f4"))
        vao = self.ctx.vertex_array(
            prog, [(buf, "3f 2f", "in_position", "in_texcoord")])
        return prog, vao

    def build_quad(self, program):
        """Return a VAO for a unit quad using the given program (needs 3f+2f)."""
        buf = self.ctx.buffer(np.array(
            [-1,-1,0,0,0, 1,-1,0,1,0, -1,1,0,0,1, 1,1,0,1,1], dtype="f4"))
        return self.ctx.vertex_array(
            program, [(buf, "3f 2f", "in_position", "in_texcoord")])

    # ── shader loader (experiment‑local) ─────────────────────
    def load_shader(self, name):
        return (self.SHADER_DIR / name).read_text()

    # ── camera ───────────────────────────────────────────────
    def cam_eye(self):
        x = self.cam_dist * math.sin(self.cam_theta) * math.cos(self.cam_phi)
        y = self.cam_dist * math.sin(self.cam_phi)
        z = self.cam_dist * math.cos(self.cam_theta) * math.cos(self.cam_phi)
        return self.cam_target + np.array([x, y, z], dtype="f4")

    def project(self, p3, vp):
        p = vp @ np.array([*p3, 1.0], dtype="f4")
        if abs(p[3]) > 1e-6: p /= p[3]
        return int((p[0]+1)*W/2), int((1-p[1])*H/2)

    # ── override points ──────────────────────────────────────
    def setup(self):                       pass
    def on_key(self, key):                 return False
    def update(self, dt):                  pass
    def render_scene(self, vp, eye):       pass
    def render_hud(self, surface, vp):     pass

    # ── event handling ───────────────────────────────────────
    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return False
            elif ev.type == pygame.KEYDOWN:
                if   ev.key == pygame.K_ESCAPE: return False
                elif ev.key == pygame.K_SPACE:  self.paused = not self.paused
                elif ev.key == pygame.K_b:      self.bloom_on = not self.bloom_on
                else: self.on_key(ev.key)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if   ev.button == 1: self._drag = True; self._last_m = ev.pos
                elif ev.button == 4: self.cam_dist = max(self.CAM_DIST_RANGE[0], self.cam_dist - 1)
                elif ev.button == 5: self.cam_dist = min(self.CAM_DIST_RANGE[1], self.cam_dist + 1)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1: self._drag = False
            elif ev.type == pygame.MOUSEMOTION and self._drag:
                dx = ev.pos[0] - self._last_m[0]
                dy = ev.pos[1] - self._last_m[1]
                self.cam_theta -= dx * 0.005
                phi_lo, phi_hi = self.CAM_PHI_RANGE
                self.cam_phi = max(phi_lo, min(phi_hi, self.cam_phi + dy * 0.005))
                self._last_m = ev.pos
        keys = pygame.key.get_pressed()
        lo, hi = self.SPEED_RANGE
        if keys[pygame.K_RIGHT]: self.speed = min(hi, self.speed + 0.002)
        if keys[pygame.K_LEFT]:  self.speed = max(lo, self.speed - 0.002)
        return True

    # ── render pipeline ──────────────────────────────────────
    def _render(self):
        eye = self.cam_eye()
        view = mat4_look_at(eye, self.cam_target, [0, 1, 0])
        proj = mat4_perspective(45.0, W / H, 0.1, 200.0)
        vp = proj @ view

        # 1. scene
        self.scene_fbo.use()
        self.ctx.clear(0, 0, 0, 1)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.stars_prog["u_time"].value = self.wall_time
        self.stars_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.render_scene(vp, eye)

        # 2. bloom
        self.ctx.disable(moderngl.DEPTH_TEST)
        if self.bloom_on:
            bw, bh = W // BLOOM_DS, H // BLOOM_DS
            self.bloom_a_fbo.use()
            self.scene_tex.use(0)
            self.bright_prog["u_texture"].value = 0
            self.bright_prog["u_threshold"].value = 0.40
            self.bright_vao.render(moderngl.TRIANGLE_STRIP)
            for sp in (1.0, 2.0):
                self.bloom_b_fbo.use()
                self.bloom_a_tex.use(0)
                self.blur_prog["u_texture"].value = 0
                self.blur_prog["u_direction"].value = (sp / bw, 0.0)
                self.blur_vao.render(moderngl.TRIANGLE_STRIP)
                self.bloom_a_fbo.use()
                self.bloom_b_tex.use(0)
                self.blur_prog["u_texture"].value = 0
                self.blur_prog["u_direction"].value = (0.0, sp / bh)
                self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        # 3. HUD
        s = self.hud_surf; s.fill((0, 0, 0, 0))
        self.render_hud(s, vp)
        self.hud_tex.write(pygame.image.tobytes(s, "RGBA", True))

        # 4. composite
        self.ctx.screen.use()
        self.ctx.clear(0, 0, 0, 1)
        self.scene_tex.use(0); self.bloom_a_tex.use(1); self.hud_tex.use(2)
        cp = self.comp_prog
        cp["u_scene"].value = 0; cp["u_bloom"].value = 1; cp["u_hud"].value = 2
        cp["u_bloom_strength"].value = 1.3
        cp["u_bloom_on"].value = int(self.bloom_on)
        self.comp_vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()

    # ── main loop ────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            dt = min(self.pg_clock.tick(60) / 1000.0, 0.05)
            running = self._handle_events()
            self.wall_time += dt
            self.update(dt)
            self._render()
        pygame.quit()
