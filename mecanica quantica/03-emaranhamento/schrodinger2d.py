"""
2D TDSE solver — Split-Step Fourier, GPU-accelerated.
Reused from experiment 02, with entangled_pair() for EPR states.
"""

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


class Schrodinger2D:

    def __init__(self, bounds, N, dt, V_func, use_gpu=True):
        self.gpu = use_gpu and _HAS_CUPY
        xp = cp if self.gpu else np

        x_min, x_max, y_min, y_max = bounds
        Nx, Ny = N
        self.Nx, self.Ny = Nx, Ny
        self.dt = dt

        self.x = np.linspace(x_min, x_max, Nx, endpoint=False)
        self.y = np.linspace(y_min, y_max, Ny, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        x_d = xp.asarray(self.x)
        y_d = xp.asarray(self.y)
        self._X, self._Y = xp.meshgrid(x_d, y_d, indexing="ij")

        kx = xp.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi
        ky = xp.fft.fftfreq(Ny, d=self.dy) * 2 * np.pi
        KX, KY = xp.meshgrid(kx, ky, indexing="ij")

        V = V_func(self._X, self._Y)
        self.V = self._to_cpu(V)
        self._exp_V_half = xp.exp(-0.5j * V * dt)
        self._exp_K = xp.exp(-0.5j * (KX**2 + KY**2) * dt)
        self._xp = xp

    def _to_cpu(self, arr):
        return arr.get() if self.gpu else arr

    def entangled_pair(self, sigma_rel, sigma_cm, k0):
        """EPR-like entangled state.

        ψ(x₁,x₂) ∝ exp(-(x₁-x₂)²/4σ_r²) · exp(-(x₁+x₂)²/4σ_R²) · exp(ik₀(x₁-x₂))

        Particle 1 (axis 0) → momentum +k₀ (right)
        Particle 2 (axis 1) → momentum −k₀ (left)
        Relative position tightly correlated (σ_r small).
        """
        xp = self._xp
        X1, X2 = self._X, self._Y
        rel = X1 - X2
        cm = X1 + X2
        psi = xp.exp(
            -rel**2 / (4 * sigma_rel**2)
            - cm**2 / (4 * sigma_cm**2)
            + 1j * k0 * rel
        )
        psi /= xp.sqrt(xp.sum(xp.abs(psi)**2) * self.dx * self.dy)
        return psi

    def step(self, psi):
        xp = self._xp
        psi = self._exp_V_half * psi
        psi = xp.fft.ifft2(self._exp_K * xp.fft.fft2(psi))
        psi = self._exp_V_half * psi
        return psi

    def evolve(self, psi, n_steps, save_every=1):
        xp = self._xp
        densities = [self._to_cpu(xp.abs(psi)**2)]
        for i in range(1, n_steps + 1):
            psi = self.step(psi)
            if i % save_every == 0:
                densities.append(self._to_cpu(xp.abs(psi)**2))
        if self.gpu:
            cp.cuda.Stream.null.synchronize()
        return self._to_cpu(psi), densities

    def norm(self, psi):
        xp = self._xp
        if isinstance(psi, np.ndarray) and self.gpu:
            psi = cp.asarray(psi)
        return float(xp.sum(xp.abs(psi)**2) * self.dx * self.dy)
