"""
01 · Lente Gravitacional — Relatividade Geral

Simulação de um buraco negro de Schwarzschild com geodésicas EXATAS.

Física implementada:
    • Equação de órbita:  d²u/dφ² + u = 3Mu²   (u = 1/r, G = c = 1)
    • Integração RK4 (800 passos) no fragment shader
    • Horizonte de eventos em r = 2M  (raio de Schwarzschild)
    • Esfera de fótons em r = 3M  (órbita circular instável)
    • ISCO em r = 6M  (última órbita estável para matéria)
    • Disco de acreção com perfil de temperatura Novikov–Thorne
    • Redshift gravitacional:  g_grav = √(1 − 3M/r)
    • Doppler relativístico:   g = g_grav / (1 − L_z Ω)
    • Intensidade:             I_obs ∝ g⁴ · B(g·T)

Controles:
    Mouse      Orbitar câmera (inclinação + azimute)
    Scroll     Zoom (distância ao BH)
    ←/→        Raio externo do disco
    ↑/↓        Escala de temperatura (cor do disco)
    D          Disco on/off
    B          Bloom on/off
    R          Reset câmera
    ESC        Sair
"""

import math
import sys
from pathlib import Path

import numpy as np
import pygame
import moderngl

# common engine (em relatividade restrita/)
_project = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project / "relatividade restrita"))
from common.engine import BaseApp, load_common, gl_bytes, W, H

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BlackHoleLens(BaseApp):
    TITLE      = "01 · Lente Gravitacional — Relatividade Geral"
    SHADER_DIR = Path(__file__).parent / "shaders"

    # câmera — inclinação alta para ver o disco e o lensing
    CAM_TARGET = (0.0, 0.0, 0.0)
    CAM_PHI    = 1.25          # ~72° acima do plano do disco
    CAM_DIST   = 50.0
    SPEED_RANGE = (0.0, 0.0)   # sem controle de velocidade neste exp.

    # ── setup ─────────────────────────────────────────────
    def setup(self):
        # programa do buraco negro (full-screen quad)
        self.bh_prog = self.ctx.program(
            vertex_shader=load_common("screen.vert"),
            fragment_shader=self.load_shader("blackhole.frag"),
        )
        fsq = np.array(
            [-1, -1, 0, 0,  1, -1, 1, 0,  -1, 1, 0, 1,  1, 1, 1, 1],
            dtype="f4",
        )
        buf = self.ctx.buffer(fsq)
        self.bh_vao = self.ctx.vertex_array(
            self.bh_prog, [(buf, "2f 2f", "in_position", "in_texcoord")]
        )

        # parâmetros físicos
        self.bh_M        = 1.0       # massa  (rs = 2)
        self.disk_inner  = 6.0       # ISCO
        self.disk_outer  = 20.0
        self.temp_scale  = 8000.0    # escala de temperatura (K)
        self.show_disk   = True

    # ── keys ──────────────────────────────────────────────
    def on_key(self, key):
        if key == pygame.K_d:
            self.show_disk = not self.show_disk
            return True
        if key == pygame.K_r:
            self.cam_phi  = 1.25
            self.cam_theta = 0.0
            self.cam_dist = 50.0
            self.disk_outer = 20.0
            self.temp_scale = 8000.0
            return True
        return False

    # ── override zoom limits ──────────────────────────────
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
                elif ev.button == 4: self.cam_dist = max(5, self.cam_dist - 2)
                elif ev.button == 5: self.cam_dist = min(500, self.cam_dist + 2)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1: self._drag = False
            elif ev.type == pygame.MOUSEMOTION and self._drag:
                dx = ev.pos[0] - self._last_m[0]
                dy = ev.pos[1] - self._last_m[1]
                self.cam_theta -= dx * 0.005
                self.cam_phi = max(0.05, min(1.55, self.cam_phi + dy * 0.005))
                self._last_m = ev.pos
        return True

    # ── update ────────────────────────────────────────────
    def update(self, dt):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            self.disk_outer = min(40.0, self.disk_outer + 8.0 * dt)
        if keys[pygame.K_LEFT]:
            self.disk_outer = max(self.disk_inner + 0.5, self.disk_outer - 8.0 * dt)
        if keys[pygame.K_UP]:
            self.temp_scale = min(30000.0, self.temp_scale + 4000.0 * dt)
        if keys[pygame.K_DOWN]:
            self.temp_scale = max(2000.0, self.temp_scale - 4000.0 * dt)

    # ── render scene ──────────────────────────────────────
    def render_scene(self, vp, eye):
        # câmera
        fwd = self.cam_target - eye
        fwd = fwd / np.linalg.norm(fwd)
        world_up = np.array([0.0, 1.0, 0.0], dtype="f4")
        right = np.cross(fwd, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype="f4")
        else:
            right = right / rn
        up = np.cross(right, fwd)

        # FOV = 45°, half-angle em radianos
        half_fov = math.radians(22.5)

        # desenhar o BH (full-screen, substitui tudo)
        self.ctx.disable(moderngl.DEPTH_TEST)

        p = self.bh_prog
        p["u_cam_pos"].value   = tuple(eye.tolist())
        p["u_cam_fwd"].value   = tuple(fwd.tolist())
        p["u_cam_right"].value = tuple(right.tolist())
        p["u_cam_up"].value    = tuple(up.tolist())
        p["u_fov"].value       = half_fov
        p["u_aspect"].value    = float(W) / float(H)

        p["u_M"].value          = self.bh_M
        p["u_disk_inner"].value = self.disk_inner if self.show_disk else 1e6
        p["u_disk_outer"].value = self.disk_outer if self.show_disk else 1e6
        p["u_temp_scale"].value = self.temp_scale
        if "u_time" in p:
            p["u_time"].value = self.wall_time

        self.bh_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)

    # ── HUD ───────────────────────────────────────────────
    def render_hud(self, s, vp):
        white = (230, 230, 240)
        dim   = (120, 120, 150)
        cyan  = (80, 190, 255)
        amber = (255, 200, 60)
        green = (50, 255, 130)

        # título
        s.blit(self.font_title.render(
            "LENTE GRAVITACIONAL", True, white), (20, 10))
        s.blit(self.font_sm.render(
            "Relatividade Geral — Schwarzschild", True, dim), (20, 42))

        # parâmetros físicos
        M  = self.bh_M
        rs = 2.0 * M
        r_photon = 3.0 * M
        r_isco   = 6.0 * M
        b_crit   = math.sqrt(27.0) * M

        params = [
            f"M = {M:.1f}     rs = {rs:.1f}",
            f"Esfera de fotons: r = {r_photon:.1f}",
            f"ISCO: r = {r_isco:.1f}",
            f"b_crit = sqrt(27)M = {b_crit:.3f}",
            "",
            f"Disco: {self.disk_inner:.0f} < r < {self.disk_outer:.0f}",
            f"T_escala = {self.temp_scale:.0f} K",
            f"Inclinacao = {math.degrees(self.cam_phi):.1f} deg",
            f"Distancia = {self.cam_dist:.1f} M",
        ]
        for i, ln in enumerate(params):
            s.blit(self.font_sm.render(ln, True, cyan), (20, 72 + i * 20))

        # controles
        ctrls = [
            "Mouse  Orbitar",
            "Scroll Zoom",
            "Setas  Disco / Temp",
            "D      Disco on/off",
            "B      Bloom",
            "R      Reset",
        ]
        for i, ln in enumerate(ctrls):
            s.blit(self.font_sm.render(ln, True, dim), (W - 240, 10 + i * 20))

        # equação
        eq_lines = [
            "Geodesica exata (RK4, 800 passos):",
            "d2u/dphi2 + u = 3Mu2",
            "b = r sin(a) / sqrt(1 - rs/r)",
            "g = sqrt(1-3M/r) / (1 - Lz * Omega)",
        ]
        for i, ln in enumerate(eq_lines):
            s.blit(self.font_sm.render(ln, True, green), (20, H - 100 + i * 20))

        # nota sobre cor
        note = "Cor: escala visivel (T real seria UV/raios-X)"
        t = self.font_sm.render(note, True, amber)
        s.blit(t, (W // 2 - t.get_width() // 2, H - 25))


if __name__ == "__main__":
    BlackHoleLens().run()
