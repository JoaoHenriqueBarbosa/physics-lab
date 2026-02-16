"""
Entanglement visualizations — three videos:

1. evolution.mp4      — |ψ(x₁,x₂)|² heatmap + marginals P(x₁), P(x₂)
2. correlations.mp4   — 2M particle accumulation revealing entanglement
3. collapse.mp4       — measuring x₁ collapses x₂: sweep + conditional
"""

import subprocess
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

_CMAP = colormaps["inferno"]
_N = 1024          # heatmap pixel size (= grid size)
_M = 120           # marginal strip thickness
_OUT = 2048        # final output size

try:
    _F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
    _Fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
except OSError:
    _F = ImageFont.load_default()
    _Fs = _F

_CYAN = np.array([0, 229, 255], dtype=np.uint8)
_GREY = np.array([60, 60, 60], dtype=np.uint8)
_RED  = np.array([255, 90, 90], dtype=np.uint8)
_MARG = np.array([41, 182, 246], dtype=np.uint8)


# ── helpers ──────────────────────────────────────────────

def _ffmpeg(path, w, h, fps=25):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path],
        stdin=subprocess.PIPE)


def _d2rgb(density, vmax, gamma=0.4):
    normed = np.clip((density / max(vmax, 1e-30)) ** gamma, 0, 1)
    rgba = _CMAP(normed.T[::-1])
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _upscale(rgb, size):
    return np.array(Image.fromarray(rgb).resize((size, size), Image.LANCZOS))


def _text(rgb, txt, pos=(12, 10), font=None):
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    f = font or _F
    d.text((pos[0]+1, pos[1]+1), txt, fill=(0, 0, 0), font=f)
    d.text(pos, txt, fill=(255, 255, 255), font=f)
    return np.array(img)


def _strip_bottom(prob, W, H, color=_MARG):
    """P(x₁) filled from top down (top touches heatmap)."""
    s = np.full((H, W, 3), 14, dtype=np.uint8)
    r = np.interp(np.linspace(0, 1, W), np.linspace(0, 1, len(prob)), prob)
    pk = r.max()
    if pk > 0:
        r /= pk
    fill = (r * (H - 4)).astype(int)
    rows = np.arange(H)[:, None]
    s[(rows >= 2) & (rows < 2 + fill[None, :])] = color
    return s


def _strip_right(prob, W, H, color=_MARG):
    """P(x₂) filled from left (left touches heatmap). y flipped to match."""
    s = np.full((H, W, 3), 14, dtype=np.uint8)
    r = np.interp(np.linspace(0, 1, H), np.linspace(0, 1, len(prob)), prob)[::-1]
    pk = r.max()
    if pk > 0:
        r /= pk
    fill = (r * (W - 4)).astype(int)
    cols = np.arange(W)[None, :]
    s[(cols >= 2) & (cols < 2 + fill[:, None])] = color
    return s


def _strip_right_dual(prob_marg, prob_cond, W, H):
    """Right strip with marginal (grey) + conditional (cyan) overlaid."""
    s = np.full((H, W, 3), 14, dtype=np.uint8)
    mx = max(prob_marg.max(), prob_cond.max(), 1e-30)
    for prob, color in [(prob_marg, _GREY), (prob_cond, _CYAN)]:
        r = np.interp(np.linspace(0, 1, H), np.linspace(0, 1, len(prob)), prob)[::-1]
        fill = (r / mx * (W - 4)).astype(int)
        cols = np.arange(W)[None, :]
        s[(cols >= 2) & (cols < 2 + fill[:, None])] = color
    return s


def _diagonal(rgb, N):
    """Faint x₁=x₂ reference line."""
    idx = np.arange(N)
    rgb[N - 1 - idx, idx] = (
        0.6 * rgb[N - 1 - idx, idx].astype(float) + 0.4 * 55
    ).astype(np.uint8)


def _measurement_line(rgb, col, N):
    """Bright red vertical line at column col."""
    lo, hi = max(col - 1, 0), min(col + 2, N)
    rgb[:, lo:hi] = (
        0.3 * rgb[:, lo:hi].astype(float) + 0.7 * _RED.astype(float)
    ).astype(np.uint8)


def _compose(heatmap, prob_x1, prob_x2, dy, dx, right_fn=None):
    """Heatmap + bottom marginal + right marginal → square frame."""
    N, M = _N, _M
    bottom = _strip_bottom(prob_x1, N, M)
    if right_fn:
        right = right_fn(M, N)
    else:
        right = _strip_right(prob_x2, M, N)
    corner = np.full((M, M, 3), 14, dtype=np.uint8)
    top = np.concatenate([heatmap, right], axis=1)
    bot = np.concatenate([bottom, corner], axis=1)
    return np.concatenate([top, bot], axis=0)


# ── 1. Evolution ────────────────────────────────────────

def render_evolution(x, y, densities, dx, dy, dt, save_every, path, fps=25):
    vmax = max(d.max() for d in densities)
    size = _OUT
    proc = _ffmpeg(path, size, size, fps)

    for idx, dens in enumerate(densities):
        hm = _d2rgb(dens, vmax)
        _diagonal(hm, _N)
        p1 = dens.sum(axis=1) * dy
        p2 = dens.sum(axis=0) * dx
        frame = _compose(hm, p1, p2, dy, dx)
        frame = _upscale(frame, size)
        t = idx * save_every * dt
        frame = _text(frame, f"t = {t:.2f}")
        frame = _text(frame, "x\u2081 (Part\u00edcula 1) \u2192",
                      pos=(size // 2 - 120, size - 55), font=_Fs)
        frame = _text(frame, "\u2191 x\u2082 (Part\u00edcula 2)",
                      pos=(size - 350, size // 2), font=_Fs)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"Salvo: {path}")


# ── 2. Correlations ─────────────────────────────────────

def render_correlations(x, y, psi_final, dx, dy,
                        n_particles, ppf, path, fps=25):
    Nx, Ny = psi_final.shape
    prob = np.abs(psi_final)**2
    pf = prob.ravel()
    pf /= pf.sum()

    print(f"  Amostrando {n_particles:,} partículas...", end=" ", flush=True)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(pf), size=n_particles, p=pf)
    ix, iy = np.unravel_index(indices, prob.shape)
    print("ok")

    n_frames = n_particles // ppf
    counts = np.zeros((Nx, Ny), dtype=np.float64)
    size = _OUT
    proc = _ffmpeg(path, size, size, fps)

    for frame in range(n_frames):
        s, e = frame * ppf, (frame + 1) * ppf
        np.add.at(counts, (ix[s:e], iy[s:e]), 1)

        vmax = counts.max() if counts.max() > 0 else 1.0
        hm = _d2rgb(counts, vmax)
        _diagonal(hm, _N)
        p1 = counts.sum(axis=1)
        p2 = counts.sum(axis=0)
        fr = _compose(hm, p1, p2, dy, dx)
        fr = _upscale(fr, size)
        fr = _text(fr, f"N = {e:,}")
        proc.stdin.write(fr.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"Salvo: {path}")


# ── 3. Collapse ─────────────────────────────────────────

def render_collapse(x, y, density_final, dx, dy, path, fps=25):
    """Sweep x₁ measurement; show conditional P(x₂|x₁*)."""
    N = _N
    prob_x1 = density_final.sum(axis=1) * dy
    prob_x2 = density_final.sum(axis=0) * dx
    vmax = density_final.max()

    # Sweep range: where P(x₁) is significant
    thresh = prob_x1.max() * 0.005
    valid = np.where(prob_x1 > thresh)[0]
    i_lo, i_hi = valid[0], valid[-1]

    n_frames = 300
    sweep = np.linspace(i_lo, i_hi, n_frames).astype(int)

    size = _OUT
    proc = _ffmpeg(path, size, size, fps)

    for i_x1 in sweep:
        hm = _d2rgb(density_final, vmax)
        _diagonal(hm, N)
        _measurement_line(hm, i_x1, N)

        # Conditional distribution P(x₂ | x₁ = x[i_x1])
        cond = density_final[i_x1, :]
        cond_norm = cond / max(cond.sum() * dy, 1e-30)

        # Bottom strip with marker
        bstrip = _strip_bottom(prob_x1, N, _M)
        marker = min(max(i_x1, 1), N - 2)
        bstrip[:, marker - 1:marker + 2] = _RED

        # Right strip: marginal grey + conditional cyan
        rstrip = _strip_right_dual(prob_x2, cond_norm, _M, N)

        corner = np.full((_M, _M, 3), 14, dtype=np.uint8)
        top = np.concatenate([hm, rstrip], axis=1)
        bot = np.concatenate([bstrip, corner], axis=1)
        fr = np.concatenate([top, bot], axis=0)
        fr = _upscale(fr, size)

        x1_val = x[i_x1]
        # Peak of conditional
        i_peak = np.argmax(cond)
        x2_val = y[i_peak]
        fr = _text(fr, f"Medi\u00e7\u00e3o: x\u2081 = {x1_val:+.1f}")
        fr = _text(fr, f"Colapso:  x\u2082 \u2248 {x2_val:+.1f}",
                   pos=(12, 35))
        proc.stdin.write(fr.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"Salvo: {path}")
