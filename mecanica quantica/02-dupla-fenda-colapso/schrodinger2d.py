"""
2D Time-Dependent Schrödinger Equation — Split-Step Fourier Method.

Natural units: ℏ = 1, m = 1.

Supports GPU acceleration via CuPy (auto-detected).
"""

import numpy as np

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


class Schrodinger2D:

    def __init__(self, bounds, N, dt, V_func, use_gpu=True):
        """
        Parameters
        ----------
        bounds   : (x_min, x_max, y_min, y_max)
        N        : (Nx, Ny)
        dt       : time step
        V_func   : callable(X, Y) → potential array (works with numpy or cupy)
        use_gpu  : attempt GPU acceleration (falls back to CPU silently)
        """
        self.gpu = use_gpu and _HAS_CUPY
        xp = cp if self.gpu else np

        x_min, x_max, y_min, y_max = bounds
        Nx, Ny = N
        self.Nx, self.Ny = Nx, Ny
        self.dt = dt

        # Keep CPU copies for visualization
        self.x = np.linspace(x_min, x_max, Nx, endpoint=False)
        self.y = np.linspace(y_min, y_max, Ny, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Build grids on the target device
        x_d = xp.asarray(self.x)
        y_d = xp.asarray(self.y)
        X, Y = xp.meshgrid(x_d, y_d, indexing="ij")

        kx = xp.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi
        ky = xp.fft.fftfreq(Ny, d=self.dy) * 2 * np.pi
        KX, KY = xp.meshgrid(kx, ky, indexing="ij")

        V = V_func(X, Y)
        self.V = self._to_cpu(V)  # CPU copy for rendering
        self._exp_V_half = xp.exp(-0.5j * V * dt)
        self._exp_K = xp.exp(-0.5j * (KX**2 + KY**2) * dt)

        self._X = X
        self._Y = Y
        self._xp = xp

    def _to_cpu(self, arr):
        return arr.get() if self.gpu else arr

    def gaussian_packet(self, x0, y0, sigma_x, sigma_y, kx0, ky0=0.0):
        xp = self._xp
        X, Y = self._X, self._Y
        psi = xp.exp(
            -(X - x0) ** 2 / (4 * sigma_x**2)
            - (Y - y0) ** 2 / (4 * sigma_y**2)
            + 1j * (kx0 * X + ky0 * Y)
        )
        psi /= xp.sqrt(xp.sum(xp.abs(psi) ** 2) * self.dx * self.dy)
        return psi

    def step(self, psi):
        xp = self._xp
        psi = self._exp_V_half * psi
        psi = xp.fft.ifft2(self._exp_K * xp.fft.fft2(psi))
        psi = self._exp_V_half * psi
        return psi

    def evolve(self, psi, n_steps, save_every=1):
        """Returns (final_psi_cpu, density_frames_cpu)."""
        xp = self._xp
        densities = [self._to_cpu(xp.abs(psi) ** 2)]
        for i in range(1, n_steps + 1):
            psi = self.step(psi)
            if i % save_every == 0:
                densities.append(self._to_cpu(xp.abs(psi) ** 2))
        if self.gpu:
            cp.cuda.Stream.null.synchronize()
        return self._to_cpu(psi), densities

    def norm(self, psi):
        xp = self._xp
        if isinstance(psi, np.ndarray) and self.gpu:
            psi = cp.asarray(psi)
        return float(xp.sum(xp.abs(psi) ** 2) * self.dx * self.dy)
