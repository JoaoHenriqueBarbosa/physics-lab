"""
01 · Dilatação Temporal — Relatividade Restrita

Dois relógios analógicos procedurais renderizados via fragment shader:
  - Esquerdo (ciano): referencial estacionário, marca tempo coordenado t
  - Direito (âmbar):  referencial em movimento, marca tempo próprio τ = t/γ

O fator de Lorentz γ = 1/√(1 − v²/c²) faz o relógio em movimento
atrasar em relação ao estacionário.

Controles:
    ←/→        Ajustar velocidade (v/c)
    SPACE      Pausar / continuar
    R          Resetar tempos
    B          Ligar/desligar bloom
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
BLOOM_DOWNSCALE = 4
SHADER_DIR = Path(__file__).parent / "shaders"

# colors (linear‑space)
COLOR_CYAN  = (0.0, 0.75, 1.0)
COLOR_AMBER = (1.0, 0.68, 0.0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matrix helpers (row‑major; transpose before uploading to GL)
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
    eye = np.asarray(eye, dtype="f4")
    target = np.asarray(target, dtype="f4")
    up = np.asarray(up, dtype="f4")
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype="f4")
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -s.dot(eye)
    m[1, 3] = -u.dot(eye)
    m[2, 3] = f.dot(eye)
    return m


def mat4_translate(tx, ty, tz):
    m = np.eye(4, dtype="f4")
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def mat4_scale(sx, sy, sz):
    m = np.eye(4, dtype="f4")
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def mat4_rotate_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    m = np.eye(4, dtype="f4")
    m[0, 0] = c;  m[0, 2] = s
    m[2, 0] = -s; m[2, 2] = c
    return m


def _gl(m):
    """Row‑major numpy → column‑major bytes for GL."""
    return np.ascontiguousarray(m.T).tobytes()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shader loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load(name):
    return (SHADER_DIR / name).read_text()


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
        pygame.display.set_caption("01 · Dilatação Temporal — Relatividade Restrita")

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._build_programs()
        self._build_geometry()
        self._build_fbos()
        self._build_hud()

        # ── physics state ──
        self.speed = 0.0          # v/c  ∈ [0, 0.995]
        self.coord_time = 0.0     # t  (estacionário)
        self.proper_time = 0.0    # τ  (em movimento)
        self.platform_x = 5.0
        self.platform_dir = 1.0
        self.paused = False
        self.bloom_on = True
        self.wall_time = 0.0

        # ── camera (spherical) ──
        self.cam_theta = 0.0      # horizontal angle
        self.cam_phi = 0.35       # vertical angle (rad)
        self.cam_dist = 20.0
        self.dragging = False
        self.last_mouse = (0, 0)

        self.clock = pygame.time.Clock()

    # ── programs ─────────────────────────────────────────────
    def _build_programs(self):
        scene_v = _load("scene.vert")
        screen_v = _load("screen.vert")

        self.grid_prog = self.ctx.program(
            vertex_shader=scene_v, fragment_shader=_load("grid.frag")
        )
        self.clock_prog = self.ctx.program(
            vertex_shader=scene_v, fragment_shader=_load("clock.frag")
        )
        self.stars_prog = self.ctx.program(
            vertex_shader=screen_v, fragment_shader=_load("stars.frag")
        )
        self.bright_prog = self.ctx.program(
            vertex_shader=screen_v, fragment_shader=_load("bloom_extract.frag")
        )
        self.blur_prog = self.ctx.program(
            vertex_shader=screen_v, fragment_shader=_load("bloom_blur.frag")
        )
        self.composite_prog = self.ctx.program(
            vertex_shader=screen_v, fragment_shader=_load("composite.frag")
        )

    # ── geometry ─────────────────────────────────────────────
    def _build_geometry(self):
        # full‑screen quad (position xy + texcoord)
        fsq = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype="f4")
        self.fsq_buf = self.ctx.buffer(fsq)

        def _screen_vao(prog):
            return self.ctx.vertex_array(
                prog, [(self.fsq_buf, "2f 2f", "in_position", "in_texcoord")]
            )

        self.stars_vao = _screen_vao(self.stars_prog)
        self.bright_vao = _screen_vao(self.bright_prog)
        self.blur_vao = _screen_vao(self.blur_prog)
        self.composite_vao = _screen_vao(self.composite_prog)

        # ground grid (position only — texcoord optimised out by grid shader)
        S = 60.0
        grid = np.array([
            -S, 0, -S,
             S, 0, -S,
            -S, 0,  S,
             S, 0,  S,
        ], dtype="f4")
        grid_buf = self.ctx.buffer(grid)
        self.grid_vao = self.ctx.vertex_array(
            self.grid_prog, [(grid_buf, "3f", "in_position")]
        )

        # clock quad (XY plane, unit square → scaled by model matrix)
        cq = np.array([
            -1, -1, 0, 0, 0,
             1, -1, 0, 1, 0,
            -1,  1, 0, 0, 1,
             1,  1, 0, 1, 1,
        ], dtype="f4")
        clock_buf = self.ctx.buffer(cq)
        self.clock_vao = self.ctx.vertex_array(
            self.clock_prog, [(clock_buf, "3f 2f", "in_position", "in_texcoord")]
        )

    # ── frame buffer objects ─────────────────────────────────
    def _build_fbos(self):
        # scene (HDR)
        self.scene_tex = self.ctx.texture((W, H), 4, dtype="f2")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_depth = self.ctx.depth_renderbuffer((W, H))
        self.scene_fbo = self.ctx.framebuffer(self.scene_tex, self.scene_depth)

        # bloom ping‑pong (lower res)
        bw, bh = W // BLOOM_DOWNSCALE, H // BLOOM_DOWNSCALE
        self.bloom_a_tex = self.ctx.texture((bw, bh), 4, dtype="f2")
        self.bloom_b_tex = self.ctx.texture((bw, bh), 4, dtype="f2")
        for t in (self.bloom_a_tex, self.bloom_b_tex):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bloom_a_fbo = self.ctx.framebuffer(self.bloom_a_tex)
        self.bloom_b_fbo = self.ctx.framebuffer(self.bloom_b_tex)

    # ── HUD text ─────────────────────────────────────────────
    def _build_hud(self):
        self.font_title = pygame.font.SysFont("monospace", 30, bold=True)
        self.font_med = pygame.font.SysFont("monospace", 22)
        self.font_sm = pygame.font.SysFont("monospace", 18)
        self.hud_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        self.hud_tex = self.ctx.texture((W, H), 4)
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ── helpers ──────────────────────────────────────────────
    @property
    def gamma(self):
        v2 = self.speed * self.speed
        return 1.0 / math.sqrt(1.0 - v2) if v2 < 1.0 else float("inf")

    def _cam_eye(self):
        x = self.cam_dist * math.sin(self.cam_theta) * math.cos(self.cam_phi)
        y = self.cam_dist * math.sin(self.cam_phi)
        z = self.cam_dist * math.cos(self.cam_theta) * math.cos(self.cam_phi)
        return np.array([x, y, z], dtype="f4")

    def _project(self, point, vp):
        p = vp @ np.array([*point, 1.0], dtype="f4")
        if abs(p[3]) > 1e-6:
            p /= p[3]
        sx = int((p[0] + 1.0) * W / 2.0)
        sy = int((1.0 - p[1]) * H / 2.0)
        return sx, sy

    # ── input ────────────────────────────────────────────────
    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                elif ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key == pygame.K_r:
                    self.coord_time = 0.0
                    self.proper_time = 0.0
                    self.platform_x = 5.0
                    self.platform_dir = 1.0
                elif ev.key == pygame.K_b:
                    self.bloom_on = not self.bloom_on
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    self.dragging = True
                    self.last_mouse = ev.pos
                elif ev.button == 4:
                    self.cam_dist = max(8.0, self.cam_dist - 1.0)
                elif ev.button == 5:
                    self.cam_dist = min(40.0, self.cam_dist + 1.0)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    self.dragging = False
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
        if self.paused:
            return

        self.coord_time += dt
        self.proper_time += dt / self.gamma

        # platform bounces between x = 2 and x = 8
        move = self.speed * 10.0 * dt * self.platform_dir
        self.platform_x += move
        if self.platform_x > 8.0:
            self.platform_x = 8.0
            self.platform_dir = -1.0
        elif self.platform_x < 2.0:
            self.platform_x = 2.0
            self.platform_dir = 1.0

    # ── render ───────────────────────────────────────────────
    def _render(self):
        eye = self._cam_eye()
        target = np.array([0.0, 3.5, 0.0], dtype="f4")
        view = mat4_look_at(eye, target, [0, 1, 0])
        proj = mat4_perspective(45.0, W / H, 0.1, 200.0)
        vp = proj @ view

        # ── 1. render scene to FBO ───────────────────────────
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # background stars
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.stars_prog["u_time"].value = self.wall_time
        self.stars_vao.render(moderngl.TRIANGLE_STRIP)

        # grid
        self.ctx.enable(moderngl.DEPTH_TEST)
        model = np.eye(4, dtype="f4")
        self.grid_prog["u_model"].write(_gl(model))
        self.grid_prog["u_viewproj"].write(_gl(vp))
        self.grid_vao.render(moderngl.TRIANGLE_STRIP)

        # clocks
        self.ctx.enable(moderngl.DEPTH_TEST)
        clock_scale = 2.5

        # stationary clock (cyan, left)
        m_static = mat4_translate(-5.0, 4.0, 0.0) @ mat4_scale(
            clock_scale, clock_scale, 1.0
        )
        self.clock_prog["u_model"].write(_gl(m_static))
        self.clock_prog["u_viewproj"].write(_gl(vp))
        self.clock_prog["u_time"].value = self.coord_time
        self.clock_prog["u_color"].value = COLOR_CYAN
        self.clock_vao.render(moderngl.TRIANGLE_STRIP)

        # moving clock (amber, right)
        m_moving = mat4_translate(self.platform_x, 4.0, 0.0) @ mat4_scale(
            clock_scale, clock_scale, 1.0
        )
        self.clock_prog["u_model"].write(_gl(m_moving))
        self.clock_prog["u_time"].value = self.proper_time
        self.clock_prog["u_color"].value = COLOR_AMBER
        self.clock_vao.render(moderngl.TRIANGLE_STRIP)

        # ── 2. bloom pass ────────────────────────────────────
        self.ctx.disable(moderngl.DEPTH_TEST)

        if self.bloom_on:
            bw = W // BLOOM_DOWNSCALE
            bh = H // BLOOM_DOWNSCALE

            # extract bright pixels
            self.bloom_a_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.scene_tex.use(0)
            self.bright_prog["u_texture"].value = 0
            self.bright_prog["u_threshold"].value = 0.45
            self.bright_vao.render(moderngl.TRIANGLE_STRIP)

            # horizontal blur → bloom_b
            self.bloom_b_fbo.use()
            self.bloom_a_tex.use(0)
            self.blur_prog["u_texture"].value = 0
            self.blur_prog["u_direction"].value = (1.0 / bw, 0.0)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)

            # vertical blur → bloom_a
            self.bloom_a_fbo.use()
            self.bloom_b_tex.use(0)
            self.blur_prog["u_texture"].value = 0
            self.blur_prog["u_direction"].value = (0.0, 1.0 / bh)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)

            # second pass for wider bloom
            self.bloom_b_fbo.use()
            self.bloom_a_tex.use(0)
            self.blur_prog["u_direction"].value = (2.0 / bw, 0.0)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)

            self.bloom_a_fbo.use()
            self.bloom_b_tex.use(0)
            self.blur_prog["u_direction"].value = (0.0, 2.0 / bh)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        # ── 3. HUD ───────────────────────────────────────────
        self._update_hud(vp)

        # ── 4. composite → screen ────────────────────────────
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.scene_tex.use(0)
        self.bloom_a_tex.use(1)
        self.hud_tex.use(2)
        self.composite_prog["u_scene"].value = 0
        self.composite_prog["u_bloom"].value = 1
        self.composite_prog["u_hud"].value = 2
        self.composite_prog["u_bloom_strength"].value = 1.2
        self.composite_prog["u_bloom_on"].value = int(self.bloom_on)
        self.composite_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()

    # ── HUD drawing ──────────────────────────────────────────
    def _update_hud(self, vp):
        s = self.hud_surf
        s.fill((0, 0, 0, 0))

        white = (230, 230, 240)
        dim = (120, 120, 150)
        cyan = (0, 200, 255)
        amber = (255, 180, 0)

        # title
        s.blit(self.font_title.render("DILATAÇÃO TEMPORAL", True, white), (20, 12))
        s.blit(
            self.font_sm.render("Relatividade Restrita", True, dim), (20, 46)
        )

        # controls
        lines = [
            "←/→   Velocidade",
            "SPACE  Pausar",
            "R      Reset",
            "B      Bloom",
            "Mouse  Orbitar",
        ]
        for i, ln in enumerate(lines):
            s.blit(self.font_sm.render(ln, True, dim), (W - 230, 12 + i * 22))

        # clock labels (projected from 3D)
        sx1, sy1 = self._project((-5.0, 6.8, 0.0), vp)
        sx2, sy2 = self._project((self.platform_x, 6.8, 0.0), vp)

        lbl1 = self.font_med.render("Estacionário", True, cyan)
        lbl2 = self.font_med.render("Em Movimento", True, amber)
        s.blit(lbl1, (sx1 - lbl1.get_width() // 2, sy1))
        s.blit(lbl2, (sx2 - lbl2.get_width() // 2, sy2))

        # time values under each clock
        st1, st1y = self._project((-5.0, 1.0, 0.0), vp)
        st2, st2y = self._project((self.platform_x, 1.0, 0.0), vp)

        t1 = self.font_med.render(f"t = {self.coord_time:.2f} s", True, cyan)
        t2 = self.font_med.render(f"τ = {self.proper_time:.2f} s", True, amber)
        s.blit(t1, (st1 - t1.get_width() // 2, st1y))
        s.blit(t2, (st2 - t2.get_width() // 2, st2y))

        # physics bar (bottom center)
        g = self.gamma
        info = [
            f"v/c = {self.speed:.4f}",
            f"γ = {g:.4f}",
            f"Δt = {self.coord_time - self.proper_time:.2f} s",
        ]
        total = "    ".join(info)
        t = self.font_med.render(total, True, white)
        s.blit(t, (W // 2 - t.get_width() // 2, H - 45))

        # speed bar
        bar_w = 300
        bar_x = W // 2 - bar_w // 2
        bar_y = H - 65
        pygame.draw.rect(s, (40, 40, 60), (bar_x, bar_y, bar_w, 8), border_radius=4)
        fill_w = int(bar_w * self.speed / 0.995)
        if fill_w > 0:
            color_interp = (
                int(255 * self.speed),
                int(200 * (1.0 - self.speed)),
                50,
            )
            pygame.draw.rect(
                s, color_interp, (bar_x, bar_y, fill_w, 8), border_radius=4
            )

        if self.paused:
            pt = self.font_title.render("PAUSADO", True, (255, 90, 90))
            s.blit(pt, (W // 2 - pt.get_width() // 2, 12))

        data = pygame.image.tobytes(s, "RGBA", True)
        self.hud_tex.write(data)

    # ── main loop ────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            dt = min(dt, 0.05)  # clamp big spikes
            running = self._handle_events()
            self._update(dt)
            self._render()
        pygame.quit()


if __name__ == "__main__":
    App().run()
