"""
03 · Transformadas de Lorentz — Relatividade Restrita

Diagrama de Minkowski inteiramente procedural (fragment shader).
Dois referenciais sobrepostos: S (azul, ortogonal) e S' (âmbar, inclinado).

Cenários (teclas 1/2/3):
  1 — Relatividade da simultaneidade
  2 — Cone de luz e causalidade
  3 — Inversão de ordem temporal

Controles:
    ←/→        Ajustar velocidade (v/c)
    1/2/3      Trocar cenário
    +/−        Zoom do diagrama
    SPACE      Pausar
    R          Resetar
    B          Bloom on/off
    Mouse      Orbitar câmera
    Scroll     Zoom câmera
    ESC        Sair
"""

import math, sys
from pathlib import Path

import numpy as np
import pygame
import moderngl

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
W, H = 1600, 900
BLOOM_DS = 4
SHADER_DIR = Path(__file__).parent / "shaders"

# ── event scenarios ──────────────────────────────────────────────
SCENARIOS = {
    1: {
        "title": "Relatividade da Simultaneidade",
        "desc": "A e B simultâneos em S (ct=3), mas não em S'",
        "events": [(2.0, 3.0), (-2.0, 3.0), (0.0, 0.0)],
        "labels": ["A", "B", "O"],
        "simul": (0, 1),   # pair connected by simultaneity line
    },
    2: {
        "title": "Cone de Luz e Causalidade",
        "desc": "A dentro do cone (timelike) · B fora (spacelike)",
        "events": [(0.0, 0.0), (1.0, 3.0), (3.5, 1.0)],
        "labels": ["O", "A", "B"],
        "simul": (-1, -1),
    },
    3: {
        "title": "Inversão de Ordem Temporal",
        "desc": "A e B spacelike — a ordem temporal inverte em S'",
        "events": [(2.5, 0.5), (-2.5, -0.5), (0.0, 0.0)],
        "labels": ["A", "B", "O"],
        "simul": (-1, -1),
    },
}

EV_COLORS_PY = [
    (50, 255, 120),   # green
    (255, 60, 75),    # red
    (90, 140, 255),   # blue
    (240, 90, 240),   # magenta
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matrix helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mat4_perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov) / 2)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect; m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
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

def _gl(m):
    return np.ascontiguousarray(m.T).tobytes()

def _load(name):
    return (SHADER_DIR / name).read_text()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class App:
    def __init__(self):
        pygame.init()
        for attr, val in [
            (pygame.GL_CONTEXT_MAJOR_VERSION, 3),
            (pygame.GL_CONTEXT_MINOR_VERSION, 3),
            (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE),
        ]:
            pygame.display.gl_set_attribute(attr, val)
        self.surface = pygame.display.set_mode((W, H), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("03 · Transformadas de Lorentz — Relatividade Restrita")

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._build_programs()
        self._build_geometry()
        self._build_fbos()
        self._build_hud()

        # state
        self.speed = 0.0
        self.display_beta = 0.0   # smoothed for animation
        self.diagram_range = 5.0
        self.scenario = 1
        self.paused = False
        self.bloom_on = True
        self.wall_time = 0.0

        # camera
        self.cam_theta = 0.0
        self.cam_phi = 0.12
        self.cam_dist = 16.0
        self.dragging = False
        self.last_mouse = (0, 0)

        self.clock = pygame.time.Clock()

    # ── programs ─────────────────────────────────────────────
    def _build_programs(self):
        sv = _load("screen.vert")
        self.diagram_prog = self.ctx.program(
            vertex_shader=_load("diagram.vert"), fragment_shader=_load("diagram.frag"))
        self.stars_prog  = self.ctx.program(vertex_shader=sv, fragment_shader=_load("stars.frag"))
        self.bright_prog = self.ctx.program(vertex_shader=sv, fragment_shader=_load("bloom_extract.frag"))
        self.blur_prog   = self.ctx.program(vertex_shader=sv, fragment_shader=_load("bloom_blur.frag"))
        self.comp_prog   = self.ctx.program(vertex_shader=sv, fragment_shader=_load("composite.frag"))

    # ── geometry ─────────────────────────────────────────────
    def _build_geometry(self):
        fsq = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype="f4")
        self.fsq_buf = self.ctx.buffer(fsq)

        def _sq(prog):
            return self.ctx.vertex_array(prog, [(self.fsq_buf, "2f 2f", "in_position", "in_texcoord")])
        self.stars_vao  = _sq(self.stars_prog)
        self.bright_vao = _sq(self.bright_prog)
        self.blur_vao   = _sq(self.blur_prog)
        self.comp_vao   = _sq(self.comp_prog)

        # diagram panel (XY plane, 10×10 units)
        S = 5.0
        dq = np.array([
            -S, -S, 0, 0, 0,
             S, -S, 0, 1, 0,
            -S,  S, 0, 0, 1,
             S,  S, 0, 1, 1,
        ], dtype="f4")
        self.diag_vao = self.ctx.vertex_array(
            self.diagram_prog, [(self.ctx.buffer(dq), "3f 2f", "in_position", "in_texcoord")])

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
        self.font_title = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_med   = pygame.font.SysFont("monospace", 20)
        self.font_sm    = pygame.font.SysFont("monospace", 16)
        self.hud_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        self.hud_tex  = self.ctx.texture((W, H), 4)
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # ── helpers ──────────────────────────────────────────────
    @property
    def gamma(self):
        b2 = self.display_beta ** 2
        return 1.0 / math.sqrt(1.0 - b2) if b2 < 1.0 else float("inf")

    def _lorentz(self, x, ct):
        g = self.gamma; b = self.display_beta
        return g * (x - b * ct), g * (ct - b * x)

    def _cam_eye(self):
        x = self.cam_dist * math.sin(self.cam_theta) * math.cos(self.cam_phi)
        y = self.cam_dist * math.sin(self.cam_phi)
        z = self.cam_dist * math.cos(self.cam_theta) * math.cos(self.cam_phi)
        return np.array([x, y, z], dtype="f4")

    def _project(self, p3, vp):
        p = vp @ np.array([*p3, 1.0], dtype="f4")
        if abs(p[3]) > 1e-6: p /= p[3]
        return int((p[0]+1)*W/2), int((1-p[1])*H/2)

    # ── events of current scenario ───────────────────────────
    @property
    def _scen(self):
        return SCENARIOS[self.scenario]

    # ── input ────────────────────────────────────────────────
    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return False
                elif ev.key == pygame.K_SPACE: self.paused = not self.paused
                elif ev.key == pygame.K_r: self.speed = 0.0
                elif ev.key == pygame.K_b: self.bloom_on = not self.bloom_on
                elif ev.key in (pygame.K_1, pygame.K_KP1): self.scenario = 1
                elif ev.key in (pygame.K_2, pygame.K_KP2): self.scenario = 2
                elif ev.key in (pygame.K_3, pygame.K_KP3): self.scenario = 3
                elif ev.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.diagram_range = max(2.0, self.diagram_range - 0.5)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.diagram_range = min(12.0, self.diagram_range + 0.5)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1: self.dragging = True; self.last_mouse = ev.pos
                elif ev.button == 4: self.cam_dist = max(6, self.cam_dist - 1)
                elif ev.button == 5: self.cam_dist = min(30, self.cam_dist + 1)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1: self.dragging = False
            elif ev.type == pygame.MOUSEMOTION and self.dragging:
                dx = ev.pos[0] - self.last_mouse[0]
                dy = ev.pos[1] - self.last_mouse[1]
                self.cam_theta -= dx * 0.005
                self.cam_phi = max(-0.5, min(1.2, self.cam_phi + dy * 0.005))
                self.last_mouse = ev.pos
        keys = pygame.key.get_pressed()
        step = 0.002
        if keys[pygame.K_RIGHT]: self.speed = min(0.95, self.speed + step)
        if keys[pygame.K_LEFT]:  self.speed = max(-0.95, self.speed - step)
        return True

    # ── update ───────────────────────────────────────────────
    def _update(self, dt):
        self.wall_time += dt
        # smooth velocity animation
        self.display_beta += (self.speed - self.display_beta) * min(dt * 6.0, 1.0)

    # ── render ───────────────────────────────────────────────
    def _render(self):
        eye = self._cam_eye()
        view = mat4_look_at(eye, [0, 0, 0], [0, 1, 0])
        proj = mat4_perspective(45, W / H, 0.1, 200)
        vp = proj @ view

        scn = self._scen
        events = scn["events"]
        simul = scn["simul"]

        # ── scene FBO ────────────────────────────────────────
        self.scene_fbo.use()
        self.ctx.clear(0, 0, 0, 1)

        # stars
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.stars_prog["u_time"].value = self.wall_time
        self.stars_vao.render(moderngl.TRIANGLE_STRIP)

        # diagram panel
        self.ctx.enable(moderngl.DEPTH_TEST)
        model = mat4_scale(1, 1, 1)  # diagram lives in XY plane
        dp = self.diagram_prog
        dp["u_model"].write(_gl(model))
        dp["u_viewproj"].write(_gl(vp))
        dp["u_beta"].value = self.display_beta
        dp["u_gamma"].value = self.gamma
        dp["u_range"].value = self.diagram_range
        dp["u_time"].value = self.wall_time
        dp["u_num_events"].value = len(events)

        # pass events
        ev_slots = ["u_ev0", "u_ev1", "u_ev2", "u_ev3"]
        for i in range(4):
            if i < len(events):
                dp[ev_slots[i]].value = events[i]
            else:
                dp[ev_slots[i]].value = (0.0, 0.0)

        dp["u_simul_a"].value = simul[0]
        dp["u_simul_b"].value = simul[1]

        self.diag_vao.render(moderngl.TRIANGLE_STRIP)

        # ── bloom ────────────────────────────────────────────
        self.ctx.disable(moderngl.DEPTH_TEST)
        if self.bloom_on:
            bw, bh = W // BLOOM_DS, H // BLOOM_DS
            self.bloom_a_fbo.use()
            self.scene_tex.use(0)
            self.bright_prog["u_texture"].value = 0
            self.bright_prog["u_threshold"].value = 0.38
            self.bright_vao.render(moderngl.TRIANGLE_STRIP)
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
        self._update_hud(vp, events, scn)

        # ── composite ────────────────────────────────────────
        self.ctx.screen.use()
        self.ctx.clear(0, 0, 0, 1)
        self.scene_tex.use(0); self.bloom_a_tex.use(1); self.hud_tex.use(2)
        cp = self.comp_prog
        cp["u_scene"].value = 0; cp["u_bloom"].value = 1; cp["u_hud"].value = 2
        cp["u_bloom_strength"].value = 1.4
        cp["u_bloom_on"].value = int(self.bloom_on)
        self.comp_vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()

    # ── HUD ──────────────────────────────────────────────────
    def _update_hud(self, vp, events, scn):
        s = self.hud_surf; s.fill((0, 0, 0, 0))
        white = (230, 230, 240); dim = (120, 120, 150)
        cyan = (80, 190, 255); amber = (255, 180, 50)

        # title
        s.blit(self.font_title.render("TRANSFORMADAS DE LORENTZ", True, white), (20, 10))
        s.blit(self.font_sm.render("Relatividade Restrita", True, dim), (20, 42))

        # scenario info
        s.blit(self.font_med.render(f"[{self.scenario}] {scn['title']}", True, (200, 200, 220)), (20, 70))
        s.blit(self.font_sm.render(scn["desc"], True, dim), (20, 95))

        # controls
        for i, ln in enumerate([
            "←/→   Velocidade", "1/2/3  Cenário",
            "+/−   Zoom diag.", "SPACE  Pausar",
            "R Reset", "B Bloom", "Mouse Orbitar",
        ]):
            s.blit(self.font_sm.render(ln, True, dim), (W - 230, 10 + i * 20))

        # axis labels (projected)
        R = self.diagram_range
        labels = [
            ((R * 0.92, 0, 0), "x", cyan),
            ((-R * 0.92, 0, 0), "−x", cyan),
            ((0, R * 0.92, 0), "ct", cyan),
            ((0, -R * 0.92, 0), "−ct", cyan),
        ]
        if abs(self.display_beta) > 0.01:
            b = self.display_beta
            # x' axis direction: (1, b, 0) normalised, scaled to range
            n = math.sqrt(1 + b * b)
            ax_len = R * 0.85
            labels.append(((ax_len / n, ax_len * b / n, 0), "x'", amber))
            labels.append(((ax_len * b / n, ax_len / n, 0), "ct'", amber))

        for p3, txt, col in labels:
            sx, sy = self._project(p3, vp)
            if 0 < sx < W and 0 < sy < H:
                t = self.font_med.render(txt, True, col)
                s.blit(t, (sx - t.get_width() // 2, sy - t.get_height() // 2))

        # event coordinate table
        g = self.gamma; b = self.display_beta
        tbl_y = H - 40 - len(events) * 24 - 30
        s.blit(self.font_med.render("Evento    (x, ct)          (x', ct')", True, white), (20, tbl_y))
        tbl_y += 26
        for i, (ex, ect) in enumerate(events):
            xp, ctp = self._lorentz(ex, ect)
            lbl = scn["labels"][i]
            ec = EV_COLORS_PY[i] if i < len(EV_COLORS_PY) else (200, 200, 200)
            line = f"  {lbl:5s} ({ex:+5.1f}, {ect:+5.1f})   →  ({xp:+5.2f}, {ctp:+5.2f})"
            s.blit(self.font_sm.render(line, True, ec), (20, tbl_y))
            tbl_y += 22

        # bottom bar
        info = f"v/c = {self.display_beta:+.4f}    γ = {g:.4f}"
        t = self.font_med.render(info, True, white)
        s.blit(t, (W // 2 - t.get_width() // 2, H - 42))

        # speed bar
        bar_w = 300; bar_x = W // 2 - 150; bar_y = H - 60
        pygame.draw.rect(s, (40, 40, 60), (bar_x, bar_y, bar_w, 8), border_radius=4)
        center = bar_w // 2
        fill = int(center * self.display_beta / 0.95)
        if fill != 0:
            rx = bar_x + center + min(fill, 0)
            rw = abs(fill)
            c_bar = (180, 120, 50)
            pygame.draw.rect(s, c_bar, (rx, bar_y, rw, 8), border_radius=4)

        if self.paused:
            pt = self.font_title.render("PAUSADO", True, (255, 90, 90))
            s.blit(pt, (W // 2 - pt.get_width() // 2, 10))

        self.hud_tex.write(pygame.image.tobytes(s, "RGBA", True))

    # ── loop ─────────────────────────────────────────────────
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
