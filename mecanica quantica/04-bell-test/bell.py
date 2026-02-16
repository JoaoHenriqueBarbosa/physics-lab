"""
Bell test physics — |Φ+⟩ state, CHSH measurement, timeline.

State: |Φ+⟩ = (|00⟩ + |11⟩) / √2
P(same)  = cos²(θ/2)      where θ = θ_a - θ_b
P(diff)  = sin²(θ/2)
E(θ)     = cos(θ)          quantum correlation
E_cl(θ)  = 1 - 2|θ|/π     classical (local hidden variable) bound
CHSH:  S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)
       with a1=0, a2=π/4, b1=π/8, b2=3π/8  →  S = 2√2 ≈ 2.83
"""

import numpy as np

# CHSH optimal angles for E(θ) = cos(θ_a - θ_b)
# S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2) = 4·cos(π/4) = 2√2
CHSH_A = np.array([0.0, np.pi / 2])             # Alice: 0°, 90°
CHSH_B = np.array([np.pi / 4, 3 * np.pi / 4])   # Bob:   45°, 135°

TOTAL_FRAMES = 1500
FPS = 25

# Phase boundaries (frame index)
PHASE1_END = 125    # 0-124:   θ=0, first measurements
PHASE2_END = 1000   # 125-999: θ sweeps 0→2π
PHASE3_END = 1375   # 1000-1374: CHSH angles, S accumulates
# 1375-1499: final hold


class BellExperiment:
    """Stateful Bell experiment accumulating measurements."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        # Accumulated scatter data: (theta, outcome_a, outcome_b)
        self.thetas = []
        self.outcomes_a = []
        self.outcomes_b = []
        # For E(θ) curve: binned correlations
        self.n_bins = 200
        self.bin_edges = np.linspace(0, 2 * np.pi, self.n_bins + 1)
        self.bin_sum = np.zeros(self.n_bins)
        self.bin_count = np.zeros(self.n_bins, dtype=int)
        # CHSH accumulator: 4 terms (a_i, b_j)
        self.chsh_sum = np.zeros(4)    # E(a_i,b_j) running sum
        self.chsh_count = np.zeros(4, dtype=int)

    def measure(self, theta_a, theta_b, n=1):
        """Perform n measurements on |Φ+⟩ at angles θ_a, θ_b.

        Returns (outcomes_a, outcomes_b) arrays of +1/-1.
        """
        theta = theta_a - theta_b
        p_same = np.cos(theta / 2) ** 2
        r = self.rng.random(n)
        same = r < p_same
        # Alice outcome: random ±1
        a = self.rng.choice([-1, 1], size=n)
        b = np.where(same, a, -a)
        return a, b

    def step(self, theta_a, theta_b, n=4):
        """Measure and accumulate results. Returns (a, b) arrays."""
        a, b = self.measure(theta_a, theta_b, n)
        theta = theta_a - theta_b
        theta_mod = theta % (2 * np.pi)
        for i in range(n):
            self.thetas.append(theta_mod)
            self.outcomes_a.append(a[i])
            self.outcomes_b.append(b[i])
            # Bin for E(θ) curve
            bi = np.searchsorted(self.bin_edges[1:], theta_mod)
            bi = min(bi, self.n_bins - 1)
            self.bin_sum[bi] += a[i] * b[i]
            self.bin_count[bi] += 1
        return a, b

    def step_chsh(self, idx_a, idx_b, n=4):
        """Measure at CHSH angle pair (idx_a, idx_b) and accumulate S."""
        theta_a = CHSH_A[idx_a]
        theta_b = CHSH_B[idx_b]
        a, b = self.measure(theta_a, theta_b, n)
        pair_idx = idx_a * 2 + idx_b
        self.chsh_sum[pair_idx] += np.sum(a * b)
        self.chsh_count[pair_idx] += n
        # Also add to scatter/bins
        theta_mod = (theta_a - theta_b) % (2 * np.pi)
        for i in range(n):
            self.thetas.append(theta_mod)
            self.outcomes_a.append(a[i])
            self.outcomes_b.append(b[i])
            bi = np.searchsorted(self.bin_edges[1:], theta_mod)
            bi = min(bi, self.n_bins - 1)
            self.bin_sum[bi] += a[i] * b[i]
            self.bin_count[bi] += 1
        return a, b

    @property
    def chsh_s(self):
        """Current CHSH S value."""
        E = np.zeros(4)
        for i in range(4):
            if self.chsh_count[i] > 0:
                E[i] = self.chsh_sum[i] / self.chsh_count[i]
        # S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)
        return E[0] - E[1] + E[2] + E[3]

    @property
    def binned_correlation(self):
        """Return (bin_centers, E_values) masking empty bins."""
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        mask = self.bin_count > 0
        E = np.full(self.n_bins, np.nan)
        E[mask] = self.bin_sum[mask] / self.bin_count[mask]
        return centers, E

    @property
    def n_measurements(self):
        return len(self.thetas)


def classical_correlation(theta):
    """E_cl(θ) = 1 - 2|θ|/π for θ ∈ [0, π], mirrored."""
    t = np.asarray(theta) % (2 * np.pi)
    t = np.where(t > np.pi, 2 * np.pi - t, t)
    return 1 - 2 * t / np.pi


def quantum_correlation(theta):
    """E_qm(θ) = cos(θ)."""
    return np.cos(theta)


class Timeline:
    """Maps frame number → experiment state for the 4-phase timeline."""

    def __init__(self):
        self.exp = BellExperiment()

    def get_state(self, frame):
        """Returns dict with all info needed to render one frame."""
        if frame < PHASE1_END:
            phase = 1
            theta_a, theta_b = 0.0, 0.0
            progress = frame / PHASE1_END
        elif frame < PHASE2_END:
            phase = 2
            t = (frame - PHASE1_END) / (PHASE2_END - PHASE1_END)
            theta_a = 0.0
            theta_b = t * 2 * np.pi
            progress = t
        elif frame < PHASE3_END:
            phase = 3
            t = (frame - PHASE2_END) / (PHASE3_END - PHASE2_END)
            theta_a, theta_b = 0.0, 0.0  # overridden by CHSH logic
            progress = t
        else:
            phase = 4
            theta_a, theta_b = 0.0, 0.0
            progress = (frame - PHASE3_END) / (TOTAL_FRAMES - PHASE3_END)

        # Perform measurements
        if phase == 1:
            a, b = self.exp.step(theta_a, theta_b, n=3)
        elif phase == 2:
            a, b = self.exp.step(theta_a, theta_b, n=4)
        elif phase == 3:
            # Cycle through CHSH angle pairs
            pair = frame % 4
            idx_a, idx_b = pair // 2, pair % 2
            theta_a = CHSH_A[idx_a]
            theta_b = CHSH_B[idx_b]
            a, b = self.exp.step_chsh(idx_a, idx_b, n=20)
        else:
            # Phase 4: no new measurements, just hold
            a = b = np.array([0])

        # Bloch sphere angles: measurement axis on each sphere
        bloch_a = theta_a
        bloch_b = theta_b
        # Last measurement results for Bloch animation
        last_a = a[-1] if len(a) > 0 else 1
        last_b = b[-1] if len(b) > 0 else 1

        centers, E_binned = self.exp.binned_correlation

        return {
            "frame": frame,
            "phase": phase,
            "progress": progress,
            "theta_a": theta_a,
            "theta_b": theta_b,
            "theta": (theta_a - theta_b) % (2 * np.pi),
            "last_a": last_a,
            "last_b": last_b,
            "bloch_a": bloch_a,
            "bloch_b": bloch_b,
            "n_measurements": self.exp.n_measurements,
            "scatter_thetas": np.array(self.exp.thetas),
            "scatter_ab": np.array(self.exp.outcomes_a) * np.array(self.exp.outcomes_b) if self.exp.thetas else np.array([]),
            "bin_centers": centers,
            "E_binned": E_binned,
            "chsh_s": self.exp.chsh_s,
            "chsh_count": self.exp.chsh_count.copy(),
        }
