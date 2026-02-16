"""
Experimento 02 — Dupla fenda: interferência + colapso
=====================================================
GPU-accelerated via CuPy + fast rendering via ffmpeg pipe.

Uso:
    python main.py          # visualização interativa (matplotlib)
    python main.py --save   # renderização rápida → propagation.mp4 + collapse.mp4
"""

import argparse
import time
import numpy as np
from schrodinger2d import Schrodinger2D
from visualize import render_propagation, render_collapse

# ── Grade espacial ───────────────────────────────────────
X_MIN, X_MAX = -80.0, 80.0
Y_MIN, Y_MAX = -80.0, 80.0
NX = NY = 1024               # 4x mais pontos (1M cells)

# ── Tempo ────────────────────────────────────────────────
DT         = 0.005
N_STEPS    = 4000            # t_max = 20.0  (2x mais tempo)
SAVE_EVERY = 20              # 200 frames de propagação

# ── Pacote de onda ───────────────────────────────────────
X0, Y0   = -40.0, 0.0       # começa mais longe → mais tempo de voo
SIGMA_X  = 5.0               # pacote mais largo
SIGMA_Y  = 15.0              # cobre bem as duas fendas
KX0      = 5.0               # momento  →  E_cin = k²/2 = 12.5

# ── Dupla fenda ──────────────────────────────────────────
WALL_X   = 0.0
WALL_W   = 0.5
SLIT_SEP = 6.0
SLIT_A   = 1.5
WALL_V0  = 1e5

# ── Colapso ──────────────────────────────────────────────
N_PARTICLES         = 2_000_000   # 2M detecções individuais
PARTICLES_PER_FRAME = 5000        # 400 frames → 16s de vídeo


def potential(X, Y):
    """Parede com duas fendas (funciona com numpy e cupy)."""
    in_wall = abs(X - WALL_X) < WALL_W / 2
    in_slit = (
        (abs(Y - SLIT_SEP / 2) < SLIT_A / 2)
        | (abs(Y + SLIT_SEP / 2) < SLIT_A / 2)
    )
    return WALL_V0 * (in_wall & ~in_slit)


def main():
    parser = argparse.ArgumentParser(description="Dupla fenda + colapso (GPU)")
    parser.add_argument("--save", action="store_true", help="salva .mp4")
    parser.add_argument("--cpu", action="store_true", help="forçar CPU")
    args = parser.parse_args()

    use_gpu = not args.cpu

    lam = 2 * np.pi / KX0
    print(f"λ = {lam:.2f}  |  fenda = {SLIT_A}  |  separação = {SLIT_SEP}")
    print(f"Grade {NX}×{NY}  |  dt = {DT}  |  t_max = {N_STEPS * DT:.1f}")

    print("Inicializando...")
    sim = Schrodinger2D(
        (X_MIN, X_MAX, Y_MIN, Y_MAX), (NX, NY), DT, potential,
        use_gpu=use_gpu,
    )
    backend = "GPU (CuPy)" if sim.gpu else "CPU (NumPy)"
    print(f"Backend: {backend}")

    psi0 = sim.gaussian_packet(X0, Y0, SIGMA_X, SIGMA_Y, KX0)

    print(f"Propagando {N_STEPS} passos...")
    t0 = time.perf_counter()
    psi_final, densities = sim.evolve(psi0, N_STEPS, SAVE_EVERY)
    sim_time = time.perf_counter() - t0

    norm = sim.norm(psi_final)
    print(f"Simulação: {sim_time:.1f}s  |  norma = {norm:.6f}  |  {len(densities)} frames")

    if args.save:
        # ── Propagação ───────────────────────────────────
        t0 = time.perf_counter()
        render_propagation(
            sim.x, sim.y, sim.V, densities, DT, SAVE_EVERY,
            "propagation.mp4",
        )
        prop_time = time.perf_counter() - t0

        # ── Colapso ─────────────────────────────────────
        t0 = time.perf_counter()
        render_collapse(
            sim.x, sim.y, sim.V, psi_final,
            N_PARTICLES, PARTICLES_PER_FRAME,
            "collapse.mp4",
        )
        coll_time = time.perf_counter() - t0

        print(f"\nTempo total: simulação {sim_time:.1f}s"
              f" + propagação {prop_time:.1f}s"
              f" + colapso {coll_time:.1f}s"
              f" = {sim_time + prop_time + coll_time:.1f}s")
    else:
        # Fallback matplotlib para visualização interativa
        from matplotlib import colormaps
        from matplotlib.colors import PowerNorm
        import matplotlib.pyplot as plt

        cmap = colormaps["inferno"]
        vmax = max(d.max() for d in densities)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            densities[-1].T, origin="lower",
            extent=[sim.x[0], sim.x[-1], sim.y[0], sim.y[-1]],
            cmap=cmap, norm=PowerNorm(gamma=0.4, vmin=0, vmax=vmax),
        )
        ax.set_title("Padrão de interferência final")
        plt.show()

    print("Concluído!")


if __name__ == "__main__":
    main()
