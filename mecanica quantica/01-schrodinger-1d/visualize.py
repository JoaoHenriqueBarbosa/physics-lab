"""Animated visualization for the 1D Schrödinger simulation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(x, V, frames, dt, save_every, save_path=None):
    """
    Two-panel animation:
      top    — |ψ(x)|²  probability density
      bottom — Re(ψ) and Im(ψ)  wave function components
    The potential V(x) is overlaid as a shaded region in both panels.
    """

    prob_max = max(np.max(np.abs(f)**2) for f in frames) * 1.15
    wave_max = max(np.max(np.abs(f)) for f in frames) * 1.15
    V_max = V.max() if V.max() > 0 else 1.0

    fig, (ax_p, ax_w) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.patch.set_facecolor("#0e0e0e")
    for ax in (ax_p, ax_w):
        ax.set_facecolor("#0e0e0e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    fig.suptitle(
        "Schrödinger 1D  —  tunelamento quântico",
        color="white", fontsize=13, y=0.96,
    )

    # ── Potential overlay ────────────────────────────────
    V_prob = V / V_max * prob_max * 0.35
    V_wave = V / V_max * wave_max * 0.30
    ax_p.fill_between(x, 0, V_prob, color="#FF9800", alpha=0.25)
    ax_p.plot(x, V_prob, color="#FF9800", lw=0.8, alpha=0.5)
    ax_w.fill_between(x, -V_wave, V_wave, color="#FF9800", alpha=0.15)

    # ── Probability panel ────────────────────────────────
    (line_prob,) = ax_p.plot([], [], color="#29B6F6", lw=1.4, label=r"$|\psi|^2$")
    ax_p.set_ylim(0, prob_max)
    ax_p.set_xlim(x[0], x[-1])
    ax_p.set_ylabel(r"$|\psi(x)|^2$", color="white")
    ax_p.legend(loc="upper right", framealpha=0.3, facecolor="#222", labelcolor="white")

    # ── Wave function panel ──────────────────────────────
    (line_re,) = ax_w.plot([], [], color="#66BB6A", lw=0.9, alpha=0.85, label=r"Re $\psi$")
    (line_im,) = ax_w.plot([], [], color="#EF5350", lw=0.9, alpha=0.85, label=r"Im $\psi$")
    ax_w.set_ylim(-wave_max, wave_max)
    ax_w.set_ylabel(r"$\psi(x)$", color="white")
    ax_w.set_xlabel("x", color="white")
    ax_w.legend(loc="upper right", framealpha=0.3, facecolor="#222", labelcolor="white")

    time_text = ax_p.text(
        0.02, 0.88, "", transform=ax_p.transAxes,
        fontsize=10, color="white", family="monospace",
    )

    def _init():
        line_prob.set_data([], [])
        line_re.set_data([], [])
        line_im.set_data([], [])
        time_text.set_text("")
        return line_prob, line_re, line_im, time_text

    def _update(idx):
        psi = frames[idx]
        line_prob.set_data(x, np.abs(psi)**2)
        line_re.set_data(x, psi.real)
        line_im.set_data(x, psi.imag)
        t = idx * save_every * dt
        time_text.set_text(f"t = {t:.2f}")
        return line_prob, line_re, line_im, time_text

    anim = FuncAnimation(
        fig, _update, init_func=_init,
        frames=len(frames), interval=30, blit=True,
    )

    if save_path:
        anim.save(save_path, fps=30, dpi=150,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        print(f"Salvo: {save_path}")
    else:
        plt.show()
