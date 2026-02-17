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

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.engine import (
    BaseApp, gl_bytes,
    mat4_scale, W, H,
)

import numpy as np
import pygame
import moderngl

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIOS = {
    1: {
        "title": "Relatividade da Simultaneidade",
        "desc": "A e B simultâneos em S (ct=3), mas não em S'",
        "events": [(2.0, 3.0), (-2.0, 3.0), (0.0, 0.0)],
        "labels": ["A", "B", "O"],
        "simul": (0, 1),
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
# Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LorentzTransforms(BaseApp):
    TITLE      = "03 · Transformadas de Lorentz — Relatividade Restrita"
    SHADER_DIR = Path(__file__).parent / "shaders"

    CAM_TARGET = (0.0, 0.0, 0.0)
    CAM_PHI    = 0.12
    CAM_DIST   = 16.0
    CAM_PHI_RANGE  = (-0.5, 1.2)
    CAM_DIST_RANGE = (6, 30)
    SPEED_RANGE = (-0.95, 0.95)

    # ── setup ──────────────────────────────────────────────────
    def setup(self):
        self.diagram_prog = self.ctx.program(
            vertex_shader=self.load_shader("diagram.vert"),
            fragment_shader=self.load_shader("diagram.frag"))
        self.diag_vao = self.build_quad(self.diagram_prog)

        # state
        self.display_beta = 0.0
        self.diagram_range = 5.0
        self.scenario = 1

    # ── helpers ────────────────────────────────────────────────
    @property
    def gamma(self):
        b2 = self.display_beta ** 2
        return 1.0 / math.sqrt(1.0 - b2) if b2 < 1.0 else float("inf")

    def _lorentz(self, x, ct):
        g = self.gamma; b = self.display_beta
        return g * (x - b * ct), g * (ct - b * x)

    @property
    def _scen(self):
        return SCENARIOS[self.scenario]

    # ── keys ───────────────────────────────────────────────────
    def on_key(self, key):
        if key == pygame.K_r:
            self.speed = 0.0
            return True
        if key in (pygame.K_1, pygame.K_KP1): self.scenario = 1; return True
        if key in (pygame.K_2, pygame.K_KP2): self.scenario = 2; return True
        if key in (pygame.K_3, pygame.K_KP3): self.scenario = 3; return True
        if key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.diagram_range = max(2.0, self.diagram_range - 0.5); return True
        if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.diagram_range = min(12.0, self.diagram_range + 0.5); return True
        return False

    # ── update ─────────────────────────────────────────────────
    def update(self, dt):
        self.display_beta += (self.speed - self.display_beta) * min(dt * 6.0, 1.0)

    # ── render scene ───────────────────────────────────────────
    def render_scene(self, vp, eye):
        scn = self._scen
        events = scn["events"]
        simul = scn["simul"]

        # diagram panel (build_quad gives -1..1, scale to -5..5)
        model = mat4_scale(5, 5, 1)
        dp = self.diagram_prog
        dp["u_model"].write(gl_bytes(model))
        dp["u_viewproj"].write(gl_bytes(vp))
        dp["u_beta"].value = self.display_beta
        dp["u_gamma"].value = self.gamma
        dp["u_range"].value = self.diagram_range
        dp["u_time"].value = self.wall_time
        dp["u_num_events"].value = len(events)

        ev_slots = ["u_ev0", "u_ev1", "u_ev2", "u_ev3"]
        for i in range(4):
            if i < len(events):
                dp[ev_slots[i]].value = events[i]
            else:
                dp[ev_slots[i]].value = (0.0, 0.0)

        dp["u_simul_a"].value = simul[0]
        dp["u_simul_b"].value = simul[1]

        self.diag_vao.render(moderngl.TRIANGLE_STRIP)

    # ── HUD ────────────────────────────────────────────────────
    def render_hud(self, s, vp):
        white = (230, 230, 240); dim = (120, 120, 150)
        cyan = (80, 190, 255); amber = (255, 180, 50)

        scn = self._scen
        events = scn["events"]

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
            n = math.sqrt(1 + b * b)
            ax_len = R * 0.85
            labels.append(((ax_len / n, ax_len * b / n, 0), "x'", amber))
            labels.append(((ax_len * b / n, ax_len / n, 0), "ct'", amber))

        for p3, txt, col in labels:
            sx, sy = self.project(p3, vp)
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


if __name__ == "__main__":
    LorentzTransforms().run()
