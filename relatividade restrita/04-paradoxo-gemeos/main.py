"""
04 · Paradoxo dos Gêmeos — Relatividade Restrita

Gêmeo A fica na Terra; Gêmeo B viaja até uma estrela distante
a velocidade v e retorna. Quando se reencontram, B é mais jovem:
    τ_B = T / γ  <  T = τ_A

A assimetria vem da mudança de referencial na meia-volta (aceleração).

Controles:
    ←/→        Ajustar velocidade (antes de iniciar)
    ENTER      Iniciar viagem
    SPACE      Pausar / continuar
    R          Resetar
    B          Bloom on/off
    Mouse      Orbitar câmera
    Scroll     Zoom
    ESC        Sair
"""

import math
import sys
from pathlib import Path

# common engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.engine import (
    BaseApp, gl_bytes, load_common,
    mat4_translate, mat4_scale, W, H,
)

import numpy as np
import pygame
import moderngl

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLOR_EARTH = (0.15, 0.45, 1.0)
COLOR_STAR  = (1.0, 0.78, 0.18)
COLOR_SHIP  = (0.0, 1.0, 0.55)

class TwinParadox(BaseApp):
    TITLE      = "04 · Paradoxo dos Gêmeos — Relatividade Restrita"
    SHADER_DIR = Path(__file__).parent / "shaders"

    CAM_TARGET = (0.0, 3.5, 0.0)
    CAM_PHI    = 0.32
    CAM_DIST   = 24.0

    # ── setup ────────────────────────────────────────────────
    def setup(self):
        # reusable scene elements
        self.grid_prog, self.grid_vao = self.build_grid()
        self.clock_prog, self.clock_vao = self.build_clock()

        # marker (glowing circle for Earth / Star / Ship)
        self.marker_prog = self.ctx.program(
            vertex_shader=load_common("world.vert"),
            fragment_shader=self.load_shader("marker.frag"),
        )
        self.marker_vao = self.build_quad(self.marker_prog)

        # layout
        self.earth_x = -6.0
        self.star_x  =  6.0
        self.D = self.star_x - self.earth_x  # 12 units

        # journey state
        self.speed = 0.60
        self._locked_v  = 0.60
        self.journey_t  = 0.0    # coordinate time elapsed
        self.proper_t   = 0.0    # proper time on ship
        self.active      = False
        self.done        = False

    # ── properties ───────────────────────────────────────────
    @property
    def gamma(self):
        v2 = self._locked_v ** 2
        return 1.0 / math.sqrt(1.0 - v2) if v2 < 1.0 else float("inf")

    @property
    def trip_duration(self):
        """Total coordinate-time round trip."""
        return self.D / self._locked_v if self._locked_v > 0 else float("inf")

    @property
    def sim_speed(self):
        """Scale so journey ≈ 10 s of real time."""
        return max(1.0, self.trip_duration / 10.0)

    @property
    def ship_x(self):
        if not self.active and not self.done:
            return self.earth_x
        T = self.trip_duration
        t = min(self.journey_t, T)
        half = T / 2.0
        v = self._locked_v
        if t <= half:
            return self.earth_x + v * t
        else:
            return self.star_x - v * (t - half)

    # ── keys ─────────────────────────────────────────────────
    def on_key(self, key):
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if not self.active and not self.done:
                self._locked_v = self.speed
                self.active = True
            return True
        if key == pygame.K_r:
            self.journey_t = 0.0
            self.proper_t  = 0.0
            self.active = False
            self.done   = False
            return True
        return False

    # ── update ───────────────────────────────────────────────
    def update(self, dt):
        if not self.active or self.paused:
            return
        step = dt * self.sim_speed
        self.journey_t += step
        self.proper_t  += step / self.gamma

        T = self.trip_duration
        if self.journey_t >= T:
            self.journey_t = T
            self.proper_t  = T / self.gamma
            self.active = False
            self.done   = True

    # ── render scene ─────────────────────────────────────────
    def render_scene(self, vp, eye):
        ident = np.eye(4, dtype="f4")

        # grid
        self.grid_prog["u_model"].write(gl_bytes(ident))
        self.grid_prog["u_viewproj"].write(gl_bytes(vp))
        self.grid_vao.render(moderngl.TRIANGLE_STRIP)

        # ── markers ──────────────────────────────────────────
        def _marker(x, y, scale, color):
            m = mat4_translate(x, y, 0) @ mat4_scale(scale, scale, 1)
            self.marker_prog["u_model"].write(gl_bytes(m))
            self.marker_prog["u_viewproj"].write(gl_bytes(vp))
            self.marker_prog["u_color"].value = color
            self.marker_vao.render(moderngl.TRIANGLE_STRIP)

        _marker(self.earth_x, 1.0, 0.9, COLOR_EARTH)   # Earth
        _marker(self.star_x,  1.0, 0.7, COLOR_STAR)     # Star
        _marker(self.ship_x,  1.0, 0.55, COLOR_SHIP)    # Ship

        # ── clocks ───────────────────────────────────────────
        cs = 2.2  # clock scale

        # Earth clock (cyan) – coordinate time
        m = mat4_translate(self.earth_x, 5.2, 0) @ mat4_scale(cs, cs, 1)
        self.clock_prog["u_model"].write(gl_bytes(m))
        self.clock_prog["u_viewproj"].write(gl_bytes(vp))
        self.clock_prog["u_time"].value = self.journey_t
        self.clock_prog["u_color"].value = (0.0, 0.6, 1.0)
        self.clock_vao.render(moderngl.TRIANGLE_STRIP)

        # Ship clock (green) – proper time
        m = mat4_translate(self.ship_x, 5.2, 0) @ mat4_scale(cs, cs, 1)
        self.clock_prog["u_model"].write(gl_bytes(m))
        self.clock_prog["u_time"].value = self.proper_t
        self.clock_prog["u_color"].value = (0.0, 0.9, 0.45)
        self.clock_vao.render(moderngl.TRIANGLE_STRIP)

    # ── HUD ──────────────────────────────────────────────────
    def render_hud(self, s, vp):
        white = (230, 230, 240)
        dim   = (120, 120, 150)
        cyan  = (80, 190, 255)
        green = (50, 255, 130)
        amber = (255, 200, 60)

        # title
        s.blit(self.font_title.render("PARADOXO DOS GÊMEOS", True, white), (20, 10))
        s.blit(self.font_sm.render("Relatividade Restrita", True, dim), (20, 42))

        # controls
        ctrls = [
            "←/→   Velocidade",
            "ENTER  Iniciar",
            "SPACE  Pausar",
            "R      Reset",
            "B      Bloom",
            "Mouse  Orbitar",
        ]
        for i, ln in enumerate(ctrls):
            s.blit(self.font_sm.render(ln, True, dim), (W - 240, 10 + i * 20))

        # 3D labels
        def _label(p3, text, color, off_y=0):
            sx, sy = self.project(p3, vp)
            t = self.font_med.render(text, True, color)
            s.blit(t, (sx - t.get_width() // 2, sy + off_y))

        _label((self.earth_x, -0.1, 0), "Terra", cyan)
        _label((self.star_x,  -0.1, 0), "Estrela", amber)

        # clock time labels
        _label((self.earth_x, 7.8, 0), f"t = {self.journey_t:.1f} s", cyan)
        _label((self.ship_x,  7.8, 0), f"τ = {self.proper_t:.1f} s", green)

        # phase label
        v_show = self._locked_v if (self.active or self.done) else self.speed
        g_show = 1.0 / math.sqrt(1 - v_show**2) if v_show < 1 else float("inf")

        if not self.active and not self.done:
            t = self.font_med.render("Ajuste v e pressione ENTER", True, (200, 200, 130))
            s.blit(t, (W // 2 - t.get_width() // 2, 70))
        elif self.active:
            phase = "ida" if self.journey_t < self.trip_duration / 2 else "volta"
            prog = min(self.journey_t / self.trip_duration, 1.0)
            t = self.font_med.render(
                f"Em viagem ({phase}) — {prog*100:.0f}%", True, green)
            s.blit(t, (W // 2 - t.get_width() // 2, 70))
        elif self.done:
            delta = self.journey_t - self.proper_t
            t = self.font_title.render(
                f"Diferença de idade: {delta:.2f} s", True, amber)
            s.blit(t, (W // 2 - t.get_width() // 2, 70))
            msg = (f"Gêmeo viajante envelheceu {self.proper_t:.1f} s  "
                   f"vs  {self.journey_t:.1f} s do estacionário")
            t2 = self.font_med.render(msg, True, white)
            s.blit(t2, (W // 2 - t2.get_width() // 2, 102))

        # bottom bar
        info = (f"v/c = {v_show:.4f}    γ = {g_show:.4f}    "
                f"D = {self.D:.0f}    T = {self.trip_duration:.1f} s")
        t = self.font_med.render(info, True, white)
        s.blit(t, (W // 2 - t.get_width() // 2, H - 42))

        # speed bar
        bar_w = 300; bar_x = W // 2 - 150; bar_y = H - 60
        pygame.draw.rect(s, (40, 40, 60), (bar_x, bar_y, bar_w, 8), border_radius=4)
        fill = int(bar_w * v_show / 0.995)
        if fill > 0:
            c_bar = (int(255 * v_show), int(200 * (1 - v_show)), 50)
            pygame.draw.rect(s, c_bar, (bar_x, bar_y, fill, 8), border_radius=4)

        if self.paused:
            t = self.font_title.render("PAUSADO", True, (255, 90, 90))
            s.blit(t, (W // 2 - t.get_width() // 2, 10))


if __name__ == "__main__":
    TwinParadox().run()
