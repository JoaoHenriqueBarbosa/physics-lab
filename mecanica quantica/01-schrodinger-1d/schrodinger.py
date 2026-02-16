"""
1D Time-Dependent Schrödinger Equation — Split-Step Fourier Method.

Natural units: ℏ = 1, m = 1.

The TDSE:  i ∂ψ/∂t = [-½ ∂²/∂x² + V(x)] ψ

Split-step scheme (2nd-order symmetric):
    ψ → exp(-i V dt/2) · ψ          half-step in position space
    ψ → FFT → exp(-i k²/2 dt) · ψ̃   full step in momentum space
    ψ → IFFT → exp(-i V dt/2) · ψ   half-step in position space
"""

import numpy as np


class Schrodinger1D:

    def __init__(self, x_min, x_max, N, dt, V_func):
        self.N = N
        self.dt = dt
        self.x = np.linspace(x_min, x_max, N, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.k = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        self.V = V_func(self.x)

        # Pre-compute exponential operators (constant potential assumed)
        self._exp_V_half = np.exp(-0.5j * self.V * dt)
        self._exp_K = np.exp(-0.5j * self.k**2 * dt)

    def gaussian_packet(self, x0, sigma, k0):
        """Gaussian wave packet centered at x0 with width sigma and momentum k0."""
        psi = np.exp(-(self.x - x0)**2 / (4 * sigma**2) + 1j * k0 * self.x)
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        return psi

    def step(self, psi):
        """Single time step via split-step Fourier."""
        psi = self._exp_V_half * psi
        psi = np.fft.ifft(self._exp_K * np.fft.fft(psi))
        psi = self._exp_V_half * psi
        return psi

    def evolve(self, psi, n_steps, save_every=1):
        """Evolve psi for n_steps, returning snapshots every save_every steps."""
        frames = [psi.copy()]
        for i in range(1, n_steps + 1):
            psi = self.step(psi)
            if i % save_every == 0:
                frames.append(psi.copy())
        return frames

    def norm(self, psi):
        """∫|ψ|² dx — should remain ≈ 1 if the grid is large enough."""
        return np.sum(np.abs(psi)**2) * self.dx
