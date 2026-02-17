<div align="center">

# ⚛ Physics Lab

**Interactive Modern Physics Simulations — from the Schrödinger equation to spacetime curvature**

GPU-accelerated · Exact Geodesics · OpenGL / GLSL · Python

<br>

<img src="assets/blackhole-side.png" width="720" alt="Schwarzschild gravitational lensing — side view with accretion disk and relativistic beaming" />

<sub>Schwarzschild gravitational lensing — 800 RK4 steps per pixel, accretion disk with relativistic Doppler and Novikov–Thorne profile</sub>

</div>

---

## About

Physics Lab is a collection of computational experiments that simulate phenomena from quantum mechanics, special relativity, and general relativity. Each experiment implements real physics — no shortcuts, no crude approximations — and renders results in real time via OpenGL/ModernGL or generates high-quality videos via ffmpeg.

The project was built around three principles: **physical fidelity** (exact equations, validated numerical methods), **performance** (GPU shaders, Split-Step FFT, CuPy when available), and **pedagogical clarity** (HUDs with real-time parameters, interactive controls to explore the parameter space).

---

## Experiments

### General Relativity

#### `01` · Gravitational Lensing — Schwarzschild

Interactive simulation of a Schwarzschild black hole with exact geodesic ray tracing. Each screen pixel fires a photon whose trajectory is integrated using the orbit equation in the Schwarzschild metric.

**Implemented physics:**

- Photon orbit equation: `d²u/dφ² + u = 3Mu²` (u = 1/r, geometric units G = c = 1)
- 4th-order Runge-Kutta integration (800 steps per ray)
- Event horizon at r = 2M, photon sphere at r = 3M, ISCO at r = 6M
- Impact parameter: `b = r₀ sin(α) / √(1 − rₛ/r₀)`
- Accretion disk with Novikov–Thorne temperature profile: `T(r) ∝ [f(r)/r³]^(1/4)`
- Gravitational redshift: `g_grav = √(1 − 3M/r)`
- Relativistic Doppler: `g = g_grav / (1 − Lz·Ω)` with `Ω = √(M/r³)`
- Observed intensity: `I_obs ∝ g⁴ · B(g·T)` (relativistic invariance)
- Blackbody → sRGB conversion (Mitchell Charity approximation)
- Procedural star field with gravitational lensing

**Controls:** Mouse (orbit), Scroll (zoom), Arrows (disk/temperature), D (disk on/off), B (bloom), R (reset)

<div align="center">
<img src="assets/blackhole-top.png" width="600" alt="Black hole seen from the pole — Einstein ring visible" />
<br>
<sub>Near-polar view (88.8°) — the Einstein ring appears as a secondary image inside the black hole shadow</sub>
</div>

---

### Quantum Mechanics

#### `01` · Quantum Tunneling — 1D Schrödinger

Gaussian wave packet incident on a rectangular potential barrier. Solves the 1D time-dependent Schrödinger equation via **Split-Step Fourier** (2nd-order symmetric):

```
ψ → exp(−iV dt/2) · ψ        half-step in position space
ψ → FFT → exp(−ik²/2 dt) · ψ̃  full step in momentum space
ψ → IFFT → exp(−iV dt/2) · ψ  half-step in position space
```

2048-point grid, natural units (ℏ = 1, m = 1). The wave packet has kinetic energy E = 12.5, the barrier has height V₀ = 15 — a classically forbidden but quantum-mechanically allowed tunneling regime.

<div align="center">
<img src="assets/tunneling-barrier.png" width="600" alt="Wave packet at the moment of tunneling" />
<br>
<sub>t = 4.80 — wave packet hitting the barrier, with part already transmitted by quantum tunneling</sub>
<br><br>
<img src="assets/tunneling-transmitted.png" width="600" alt="Reflected and transmitted packet after tunneling" />
<br>
<sub>t = 8.80 — after interaction: reflected (left) and transmitted (right) packets, demonstrating tunneling through a classically forbidden barrier</sub>
</div>

---

#### `02` · Double Slit — Interference and Collapse

2D Schrödinger on a 1024×1024 grid (1M cells), GPU-accelerated via CuPy. Simulates a wave packet passing through a double slit and forming the interference pattern. Includes wave function collapse simulation with positional detection.

<div align="center">
<img src="assets/double-slit-interference.png" width="480" alt="Double slit interference pattern" />
<br>
<sub>t = 15.0 — |ψ(x,y)|² after passing through the double slit, showing the quantum interference pattern</sub>
<br><br>
<img src="assets/double-slit-collapse.png" width="480" alt="Wave function collapse with 1M detections" />
<br>
<sub>N = 1,005,000 detections — the collapse histogram reproduces the |ψ|² interference pattern</sub>
</div>

---

#### `03` · Quantum Entanglement (EPR)

Pair of particles in an entangled state flying in opposite directions. 2D wave function: `ψ(x₁,x₂) ∝ exp(−(x₁−x₂)²/4σᵣ²) · exp(−(x₁+x₂)²/4σᵣ²) · exp(ik₀(x₁−x₂))`. 1024×1024 grid with GPU. Generates three videos: |ψ|² evolution, 2 million detections revealing correlations, and instantaneous collapse upon measuring one particle.

<div align="center">
<img src="assets/entanglement-evolution.png" width="480" alt="Entangled state evolution in x1-x2 space" />
<br>
<sub>|ψ(x₁,x₂)|² at t = 14.0 — the joint distribution in (x₁, x₂) space shows correlation along the diagonal, while the marginal distributions (projections) are broad</sub>
<br><br>
<img src="assets/entanglement-correlations.png" width="480" alt="1.7M detections revealing quantum correlation" />
<br>
<sub>N = 1,755,000 detections — the scatter plot (x₁, x₂) reveals the entanglement correlation: measuring x₁ determines x₂</sub>
<br><br>
<img src="assets/entanglement-collapse.png" width="480" alt="Instantaneous collapse by measurement" />
<br>
<sub>Measurement of x₁ = +0.2 instantaneously collapses x₂ to −43.9 — the red line marks the measurement slice in configuration space</sub>
</div>

---

#### `04` · Bell Test (CHSH)

Full Bell test simulation with the |Φ+⟩ = (|00⟩ + |11⟩)/√2 state. OpenGL rendering with 4 synchronized panels (1920×1080) showing the violation of Bell's inequality via the CHSH protocol:

```
S = E(a₁,b₁) − E(a₁,b₂) + E(a₂,b₁) + E(a₂,b₂) = 2√2 ≈ 2.83 > 2
```

The value S = 2√2 (Tsirelson's bound) violates the classical limit S ≤ 2, ruling out local hidden variable theories.

<div align="center">
<img src="assets/bell-sweep.png" width="640" alt="Bell test angular sweep" />
<br>
<sub>Phase 2: angular sweep θ — the 4 panels show EPR source, Bloch spheres (Alice and Bob), detection scatter (a×b), and correlation E(θ) vs classical and quantum predictions</sub>
<br><br>
<img src="assets/bell-chsh.png" width="640" alt="Final CHSH result S = 2.87" />
<br>
<sub>Final result: S = 2.87 > 2 — Bell inequality violation confirmed, consistent with the quantum limit of 2√2 ≈ 2.83</sub>
</div>

---

### Special Relativity

#### `01` · Time Dilation

Two procedural analog clocks rendered via fragment shader. The stationary clock marks coordinate time *t*, while the moving clock marks proper time *τ = t/γ*. The Lorentz factor `γ = 1/√(1 − v²/c²)` visually delays the moving clock's hand in real time.

#### `02` · Lorentz Contraction

A cube and a sphere side by side, with ghost wireframes showing the rest length L₀ and solid shapes showing the contracted length `L = L₀/γ`. The contraction happens entirely in the GPU vertex shader — each vertex is compressed along the axis of motion by the factor `1/γ = √(1 − v²/c²)`.

#### `03` · Lorentz Transforms

Fully procedural Minkowski diagram (fragment shader). Two overlaid reference frames: S (blue, orthogonal) and S' (amber, tilted). Three interactive scenarios: relativity of simultaneity, light cone and causality, temporal order reversal.

#### `04` · Twin Paradox

Twin A stays on Earth; Twin B travels to a distant star at velocity v and returns. When they reunite, B is younger: `τ_B = T/γ < T = τ_A`. The simulation shows the asymmetry that comes from the change of reference frame at the turnaround (acceleration), resolving the "paradox".

---

## Architecture

```
physics-lab/
├── relatividade geral/
│   └── 01-lente-gravitacional/    # Geodesic ray tracing (GLSL)
│       ├── main.py                # Interactive app (Pygame + ModernGL)
│       └── shaders/
│           └── blackhole.frag     # RK4 in fragment shader
├── relatividade restrita/
│   ├── common/                    # Shared engine
│   │   ├── engine.py              # BaseApp, bloom, orbital camera, HUD
│   │   └── shaders/               # Bloom, grid, starfield, compositing
│   ├── 01-dilatacao-temporal/
│   ├── 02-contracao-lorentz/
│   ├── 03-transformadas-lorentz/
│   └── 04-paradoxo-gemeos/
├── mecanica quantica/
│   ├── 01-schrodinger-1d/         # Split-Step Fourier 1D
│   ├── 02-dupla-fenda-colapso/    # Split-Step Fourier 2D (CuPy/GPU)
│   ├── 03-emaranhamento/          # EPR with 2M detections
│   └── 04-bell-test/              # CHSH with OpenGL renderer
└── requirements.txt
```

The shared engine (`common/engine.py`) provides: OpenGL 3.3 Core context via Pygame, HDR bloom pipeline (extract → multi-pass Gaussian blur → composite), orbital mouse camera, HUD system with procedural fonts, and full-screen quad framework for ray tracing shaders.

---

## Installation

```bash
git clone https://github.com/JoaoHenriqueBarbosa/physics-lab.git
cd physics-lab
pip install -r requirements.txt
```

**Dependencies:** NumPy, Matplotlib, Pygame, ModernGL. For GPU acceleration in 2D quantum mechanics experiments, also install [CuPy](https://cupy.dev/) (optional — works with pure NumPy, but slower).

**System requirements:** OpenGL 3.3+ (any dedicated or integrated GPU from the last ~10 years).

### Running

```bash
# General Relativity — interactive black hole
python "relatividade geral/01-lente-gravitacional/main.py"

# Special Relativity — time dilation
python "relatividade restrita/01-dilatacao-temporal/main.py"

# Quantum Mechanics — tunneling (interactive)
python "mecanica quantica/01-schrodinger-1d/main.py"

# Quantum Mechanics — double slit (generates video)
python "mecanica quantica/02-dupla-fenda-colapso/main.py" --save
```

---

## References

- Schwarzschild, K. (1916) — Schwarzschild Metric
- Luminet, J.-P. (1979) — *Image of a Spherical Black Hole with Thin Accretion Disk*, Astron. Astrophys. 75, 228–235
- Novikov, I. D. & Thorne, K. S. (1973) — *Black Holes* (Les Houches)
- James, O. et al. (2015) — *Gravitational Lensing by Spinning Black Holes in Astrophysics, and in the Movie Interstellar*, Class. Quant. Grav. 32, 065001
- Bruneton, E. (2020) — arXiv:2010.08735
- Clauser, J. F. et al. (1969) — CHSH Inequality

---

<div align="center">

Made with real physics, coffee, and one productive night.

</div>
