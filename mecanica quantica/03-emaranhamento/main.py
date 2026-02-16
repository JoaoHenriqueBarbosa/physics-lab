"""
Experimento 03 — Emaranhamento quântico (EPR)
=============================================
Par de partículas em estado emaranhado voando em direções opostas.

ψ(x₁,x₂) ∝ exp(-(x₁-x₂)²/4σ_r²) · exp(-(x₁+x₂)²/4σ_R²) · exp(ik₀(x₁-x₂))

Gera três vídeos:
  1. evolution.mp4     — |ψ(x₁,x₂)|² + marginais (partículas se separando)
  2. correlations.mp4  — 2M detecções (x₁,x₂) revelando correlação
  3. collapse.mp4      — medir x₁ colapsa x₂ instantaneamente

Uso:
    python main.py --save
    python main.py --save --cpu
"""

import argparse
import time
import numpy as np
from schrodinger2d import Schrodinger2D
from visualize import render_evolution, render_correlations, render_collapse

# ── Grade ────────────────────────────────────────────────
X_MIN, X_MAX = -80.0, 80.0
Y_MIN, Y_MAX = -80.0, 80.0
NX = NY = 1024

# ── Tempo ────────────────────────────────────────────────
DT         = 0.01
N_STEPS    = 2000           # t_max = 20
SAVE_EVERY = 20             # 100 frames

# ── Par emaranhado ───────────────────────────────────────
SIGMA_REL  = 1.5            # posição relativa apertada → emaranhamento forte
SIGMA_CM   = 10.0           # centro de massa espalhado → posições individuais incertas
K0         = 3.0            # p₁ = +k₀, p₂ = −k₀ → voam em direções opostas

# ── Correlações ──────────────────────────────────────────
N_PARTICLES         = 2_000_000
PARTICLES_PER_FRAME = 5000


def potential(X, Y):
    return X * 0.0           # partículas livres


def main():
    parser = argparse.ArgumentParser(description="Emaranhamento quântico")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    print(f"σ_rel = {SIGMA_REL}  |  σ_cm = {SIGMA_CM}  |  k₀ = {K0}")
    print(f"Razão σ_cm/σ_rel = {SIGMA_CM/SIGMA_REL:.1f} (entanglement strength)")
    print(f"Grade {NX}×{NY}  |  dt = {DT}  |  t_max = {N_STEPS * DT:.0f}")

    sim = Schrodinger2D(
        (X_MIN, X_MAX, Y_MIN, Y_MAX), (NX, NY), DT, potential,
        use_gpu=not args.cpu,
    )
    print(f"Backend: {'GPU (CuPy)' if sim.gpu else 'CPU (NumPy)'}")

    psi0 = sim.entangled_pair(SIGMA_REL, SIGMA_CM, K0)

    print(f"Propagando {N_STEPS} passos...")
    t0 = time.perf_counter()
    psi_final, densities = sim.evolve(psi0, N_STEPS, SAVE_EVERY)
    sim_t = time.perf_counter() - t0
    norm = sim.norm(psi_final)
    print(f"Simulação: {sim_t:.1f}s  |  norma = {norm:.6f}  |  {len(densities)} frames")

    if not args.save:
        print("Use --save para gerar os vídeos.")
        return

    # ── 1. Evolution ─────────────────────────────────────
    t0 = time.perf_counter()
    render_evolution(sim.x, sim.y, densities, sim.dx, sim.dy,
                     DT, SAVE_EVERY, "evolution.mp4")
    evo_t = time.perf_counter() - t0

    # ── 2. Correlations ──────────────────────────────────
    t0 = time.perf_counter()
    render_correlations(sim.x, sim.y, psi_final, sim.dx, sim.dy,
                        N_PARTICLES, PARTICLES_PER_FRAME, "correlations.mp4")
    corr_t = time.perf_counter() - t0

    # ── 3. Collapse ──────────────────────────────────────
    t0 = time.perf_counter()
    render_collapse(sim.x, sim.y, densities[-1], sim.dx, sim.dy, "collapse.mp4")
    coll_t = time.perf_counter() - t0

    total = sim_t + evo_t + corr_t + coll_t
    print(f"\nTempo: sim {sim_t:.1f}s + evo {evo_t:.1f}s"
          f" + corr {corr_t:.1f}s + colapso {coll_t:.1f}s = {total:.1f}s")
    print("Concluído!")


if __name__ == "__main__":
    main()
