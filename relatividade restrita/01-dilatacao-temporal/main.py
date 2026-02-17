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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.engine import (
    BaseApp, gl_bytes,
    mat4_translate, mat4_scale, W, H,
)

import numpy as np
import pygame
import moderngl

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLOR_CYAN  = (0.0, 0.75, 1.0)
COLOR_AMBER = (1.0, 0.68, 0.0)


class TimeDilation(BaseApp):
    TITLE      = "01 · Dilatação Temporal — Relatividade Restrita"

    CAM_TARGET = (0.0, 3.5, 0.0)
    CAM_PHI    = 0.18
    CAM_DIST   = 19.0

    # ── setup ──────────────────────────────────────────────────
    def setup(self):
        self.grid_prog, self.grid_vao = self.build_grid()
        self.clock_prog, self.clock_vao = self.build_clock()

        # physics state
        self.coord_time = 0.0
        self.proper_time = 0.0
        self.platform_x = 5.0
        self.platform_dir = 1.0

    # ── helpers ────────────────────────────────────────────────
    @property
    def gamma(self):
        v2 = self.speed * self.speed
        return 1.0 / math.sqrt(1.0 - v2) if v2 < 1.0 else float("inf")

    # ── keys ───────────────────────────────────────────────────
    def on_key(self, key):
        if key == pygame.K_r:
            self.coord_time = 0.0
            self.proper_time = 0.0
            self.platform_x = 5.0
            self.platform_dir = 1.0
            return True
        return False

    # ── update ─────────────────────────────────────────────────
    def update(self, dt):
        if self.paused:
            return

        self.coord_time += dt
        self.proper_time += dt / self.gamma

        move = self.speed * 10.0 * dt * self.platform_dir
        self.platform_x += move
        if self.platform_x > 8.0:
            self.platform_x = 8.0
            self.platform_dir = -1.0
        elif self.platform_x < 2.0:
            self.platform_x = 2.0
            self.platform_dir = 1.0

    # ── render scene ───────────────────────────────────────────
    def render_scene(self, vp, eye):
        ident = np.eye(4, dtype="f4")

        # grid
        self.grid_prog["u_model"].write(gl_bytes(ident))
        self.grid_prog["u_viewproj"].write(gl_bytes(vp))
        self.grid_vao.render(moderngl.TRIANGLE_STRIP)

        # clocks
        clock_scale = 2.5

        # stationary clock (cyan, left)
        m_static = mat4_translate(-5.0, 4.0, 0.0) @ mat4_scale(
            clock_scale, clock_scale, 1.0
        )
        self.clock_prog["u_model"].write(gl_bytes(m_static))
        self.clock_prog["u_viewproj"].write(gl_bytes(vp))
        self.clock_prog["u_time"].value = self.coord_time
        self.clock_prog["u_color"].value = COLOR_CYAN
        self.clock_vao.render(moderngl.TRIANGLE_STRIP)

        # moving clock (amber, right)
        m_moving = mat4_translate(self.platform_x, 4.0, 0.0) @ mat4_scale(
            clock_scale, clock_scale, 1.0
        )
        self.clock_prog["u_model"].write(gl_bytes(m_moving))
        self.clock_prog["u_time"].value = self.proper_time
        self.clock_prog["u_color"].value = COLOR_AMBER
        self.clock_vao.render(moderngl.TRIANGLE_STRIP)

    # ── HUD ────────────────────────────────────────────────────
    def render_hud(self, s, vp):
        white = (230, 230, 240)
        dim   = (120, 120, 150)
        cyan  = (0, 200, 255)
        amber = (255, 180, 0)

        # title
        s.blit(self.font_title.render("DILATAÇÃO TEMPORAL", True, white), (20, 12))
        s.blit(self.font_sm.render("Relatividade Restrita", True, dim), (20, 46))

        # controls
        for i, ln in enumerate([
            "←/→   Velocidade",
            "SPACE  Pausar",
            "R      Reset",
            "B      Bloom",
            "Mouse  Orbitar",
        ]):
            s.blit(self.font_sm.render(ln, True, dim), (W - 230, 12 + i * 22))

        # clock labels (projected from 3D)
        sx1, sy1 = self.project((-5.0, 6.8, 0.0), vp)
        sx2, sy2 = self.project((self.platform_x, 6.8, 0.0), vp)
        lbl1 = self.font_med.render("Estacionário", True, cyan)
        lbl2 = self.font_med.render("Em Movimento", True, amber)
        s.blit(lbl1, (sx1 - lbl1.get_width() // 2, sy1))
        s.blit(lbl2, (sx2 - lbl2.get_width() // 2, sy2))

        # time values under each clock
        st1, st1y = self.project((-5.0, 1.0, 0.0), vp)
        st2, st2y = self.project((self.platform_x, 1.0, 0.0), vp)
        t1 = self.font_med.render(f"t = {self.coord_time:.2f} s", True, cyan)
        t2 = self.font_med.render(f"τ = {self.proper_time:.2f} s", True, amber)
        s.blit(t1, (st1 - t1.get_width() // 2, st1y))
        s.blit(t2, (st2 - t2.get_width() // 2, st2y))

        # physics bar (bottom center)
        g = self.gamma
        total = (f"v/c = {self.speed:.4f}    γ = {g:.4f}    "
                 f"Δt = {self.coord_time - self.proper_time:.2f} s")
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


if __name__ == "__main__":
    TimeDilation().run()
