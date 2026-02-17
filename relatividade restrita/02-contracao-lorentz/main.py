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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.engine import (
    BaseApp, gl_bytes,
    mat4_translate, mat4_scale, mat4_rotate_y, W, H,
)

import numpy as np
import pygame
import moderngl

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLOR_CYAN  = (0.0, 0.75, 1.0)
COLOR_AMBER = (1.0, 0.68, 0.0)

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
            verts.extend([x, y, z, x, y, z])
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
    for lat_deg in range(-90 + lat_step, 90, lat_step):
        lat = math.radians(lat_deg)
        y = math.sin(lat); rc = math.cos(lat)
        for i in range(segs):
            a1 = 2.0 * math.pi * i / segs
            a2 = 2.0 * math.pi * (i + 1) / segs
            verts.extend([rc * math.cos(a1), y, rc * math.sin(a1)])
            verts.extend([rc * math.cos(a2), y, rc * math.sin(a2)])
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

class LorentzContraction(BaseApp):
    TITLE      = "02 · Contração de Lorentz — Relatividade Restrita"
    SHADER_DIR = Path(__file__).parent / "shaders"

    CAM_TARGET = (0.0, 1.8, 0.0)
    CAM_THETA  = 0.35
    CAM_PHI    = 0.38
    CAM_DIST   = 18.0
    CAM_DIST_RANGE = (8, 40)

    # ── setup ──────────────────────────────────────────────────
    def setup(self):
        self.grid_prog, self.grid_vao = self.build_grid()

        # experiment-specific programs
        self.cube_prog = self.ctx.program(
            vertex_shader=self.load_shader("cube.vert"),
            fragment_shader=self.load_shader("cube.frag"))
        self.wire_prog = self.ctx.program(
            vertex_shader=self.load_shader("wire.vert"),
            fragment_shader=self.load_shader("wire.frag"))

        # geometry
        cf = _cube_faces()
        self.cube_vao = self.ctx.vertex_array(
            self.cube_prog, [(self.ctx.buffer(cf), "3f 3f", "in_position", "in_normal")])

        ce = _cube_edges()
        self.cube_wire_vao = self.ctx.vertex_array(
            self.wire_prog, [(self.ctx.buffer(ce), "3f", "in_position")])

        sv, si = _sphere_faces()
        self.sphere_vao = self.ctx.vertex_array(
            self.cube_prog,
            [(self.ctx.buffer(sv), "3f 3f", "in_position", "in_normal")],
            index_buffer=self.ctx.buffer(si))

        sw = _sphere_wireframe()
        self.sphere_wire_vao = self.ctx.vertex_array(
            self.wire_prog, [(self.ctx.buffer(sw), "3f", "in_position")])

        # state
        self.auto_rotate = True
        self.rotation = 0.0

    # ── helpers ────────────────────────────────────────────────
    @property
    def gamma(self):
        v2 = self.speed * self.speed
        return 1.0 / math.sqrt(1.0 - v2) if v2 < 1.0 else float("inf")

    @property
    def contraction(self):
        v2 = self.speed * self.speed
        return math.sqrt(1.0 - v2) if v2 < 1.0 else 0.001

    # ── keys ───────────────────────────────────────────────────
    def on_key(self, key):
        if key == pygame.K_r:
            self.speed = 0.0; self.rotation = 0.0
            return True
        if key == pygame.K_a:
            self.auto_rotate = not self.auto_rotate
            return True
        return False

    # ── update ─────────────────────────────────────────────────
    def update(self, dt):
        if not self.paused and self.auto_rotate:
            self.rotation += dt * 0.4

    # ── render helpers ─────────────────────────────────────────
    def _set_wire(self, model, vp, contraction, motion, color, alpha):
        p = self.wire_prog
        p["u_model"].write(gl_bytes(model))
        p["u_viewproj"].write(gl_bytes(vp))
        p["u_contraction"].value = contraction
        p["u_motion_dir"].value = motion
        p["u_color"].value = color
        p["u_alpha"].value = alpha

    def _set_cube(self, model, vp, contraction, motion, color, alpha, cam):
        p = self.cube_prog
        p["u_model"].write(gl_bytes(model))
        p["u_viewproj"].write(gl_bytes(vp))
        p["u_contraction"].value = contraction
        p["u_motion_dir"].value = motion
        p["u_color"].value = color
        p["u_alpha"].value = alpha
        p["u_cam_pos"].value = tuple(cam)

    # ── render scene ───────────────────────────────────────────
    def render_scene(self, vp, eye):
        c = self.contraction
        motion = (1.0, 0.0, 0.0)
        obj_scale = 1.8

        model_cube   = mat4_translate(-4, 2.2, 0) @ mat4_rotate_y(self.rotation) @ mat4_scale(obj_scale, obj_scale, obj_scale)
        model_sphere = mat4_translate( 4, 2.2, 0) @ mat4_rotate_y(self.rotation) @ mat4_scale(obj_scale, obj_scale, obj_scale)

        # grid
        self.grid_prog["u_model"].write(gl_bytes(np.eye(4, dtype="f4")))
        self.grid_prog["u_viewproj"].write(gl_bytes(vp))
        self.grid_vao.render(moderngl.TRIANGLE_STRIP)

        # ghost wireframes (rest frame, contraction=1)
        self.ctx.depth_mask = False
        ghost_cyan  = (0.0, 0.25, 0.4)
        ghost_amber = (0.4, 0.28, 0.0)

        self._set_wire(model_cube, vp, 1.0, motion, ghost_cyan, 0.25)
        self.cube_wire_vao.render(moderngl.LINES)

        self._set_wire(model_sphere, vp, 1.0, motion, ghost_amber, 0.20)
        self.sphere_wire_vao.render(moderngl.LINES)

        # contracted solid shapes
        self._set_cube(model_cube, vp, c, motion, COLOR_CYAN, 0.28, eye)
        self.ctx.cull_face = "back"
        self.ctx.enable(moderngl.CULL_FACE)
        self.cube_vao.render(moderngl.TRIANGLES)

        self._set_cube(model_sphere, vp, c, motion, COLOR_AMBER, 0.28, eye)
        self.sphere_vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.CULL_FACE)

        # contracted wireframes (bright, on top)
        self._set_wire(model_cube, vp, c, motion, (0.0, 0.9, 1.0), 0.85)
        self.cube_wire_vao.render(moderngl.LINES)

        self._set_wire(model_sphere, vp, c, motion, (1.0, 0.72, 0.0), 0.65)
        self.sphere_wire_vao.render(moderngl.LINES)

        self.ctx.depth_mask = True

    # ── HUD ────────────────────────────────────────────────────
    def render_hud(self, s, vp):
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
        sx1, sy1 = self.project((-4, 4.5, 0), vp)
        sx2, sy2 = self.project(( 4, 4.5, 0), vp)
        lbl1 = self.font_med.render("Cubo", True, cyan)
        lbl2 = self.font_med.render("Esfera", True, amber)
        s.blit(lbl1, (sx1 - lbl1.get_width() // 2, sy1))
        s.blit(lbl2, (sx2 - lbl2.get_width() // 2, sy2))

        # dimensions below objects
        L0 = 1.8 * 2
        L = L0 * self.contraction
        st1, st1y = self.project((-4, -0.2, 0), vp)
        st2, st2y = self.project(( 4, -0.2, 0), vp)
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


if __name__ == "__main__":
    LorentzContraction().run()
