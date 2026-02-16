"""
Experimento 01 — Tunelamento quântico
======================================
Pacote de onda gaussiano incidindo em uma barreira retangular de potencial.
Resolve a equação de Schrödinger 1D dependente do tempo via Split-Step Fourier.

Unidades naturais: ℏ = 1, m = 1.

Uso:
    python main.py          # visualização interativa
    python main.py --save   # salva animação em tunneling.mp4
"""

import argparse
import numpy as np
from schrodinger import Schrodinger1D
from visualize import animate

# ── Grade espacial ───────────────────────────────────────
X_MIN, X_MAX = -80.0, 80.0
N            = 2048            # pontos (potência de 2 → FFT rápida)

# ── Tempo ────────────────────────────────────────────────
DT           = 0.005           # passo temporal
N_STEPS      = 2000            # passos totais  (t_max = 10)
SAVE_EVERY   = 8               # salva 1 frame a cada 8 passos → 250 frames

# ── Pacote de onda ───────────────────────────────────────
X0    = -15.0                  # posição inicial
SIGMA = 2.0                    # largura
K0    = 5.0                    # momento  →  E_cin = k₀²/2 = 12.5

# ── Barreira de potencial ────────────────────────────────
V0     = 15.0                  # altura (> E_cin → regime de tunelamento)
WIDTH  = 0.5                   # largura
CENTER = 0.0                   # posição


def potential(x):
    """Barreira retangular."""
    return V0 * (np.abs(x - CENTER) < WIDTH / 2).astype(float)


def main():
    parser = argparse.ArgumentParser(description="Tunelamento quântico 1D")
    parser.add_argument("--save", action="store_true", help="salva em tunneling.mp4")
    args = parser.parse_args()

    sim = Schrodinger1D(X_MIN, X_MAX, N, DT, potential)
    psi0 = sim.gaussian_packet(X0, SIGMA, K0)

    E_kin = K0**2 / 2
    print(f"E_cin = {E_kin:.1f}  |  V_barreira = {V0}  |  regime: ", end="")
    print("tunelamento" if V0 > E_kin else "transmissão parcial")
    print(f"Simulando {N_STEPS} passos (dt={DT}, t_max={N_STEPS*DT:.1f}) ...")

    frames = sim.evolve(psi0, N_STEPS, SAVE_EVERY)

    # Checar conservação da norma
    norm_final = sim.norm(frames[-1])
    print(f"Norma final: {norm_final:.6f}  (desvio: {abs(1 - norm_final):.2e})")
    print(f"{len(frames)} frames capturados. Animando...")

    save_path = "tunneling.mp4" if args.save else None
    animate(sim.x, sim.V, frames, DT, SAVE_EVERY, save_path)


if __name__ == "__main__":
    main()
