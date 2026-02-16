"""
Fast rendering pipeline — bypasses matplotlib, pipes raw RGB frames to ffmpeg.

Propagation:  colormap + wall overlay → ffmpeg
Collapse:     incremental dot canvas  → ffmpeg
"""

import subprocess
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

# ── Configuration ────────────────────────────────────────

_CMAP = colormaps["inferno"]
_BG = np.array([14, 14, 14], dtype=np.uint8)  # #0e0e0e
_WALL_COLOR = np.array([115, 115, 115], dtype=np.uint8)
_DOT_COLOR = np.array([0, 229, 255], dtype=np.uint8)  # cyan
_OUTPUT_SIZE = 2048  # upscale to this resolution
_DOT_RADIUS = 2      # 5x5 pixel dots at output resolution

try:
    _FONT = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18,
    )
except OSError:
    _FONT = ImageFont.load_default()


# ── Helpers ──────────────────────────────────────────────

def _open_ffmpeg(path, w, h, fps=20):
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            path,
        ],
        stdin=subprocess.PIPE,
    )


def _density_to_rgb(density, vmax, gamma=0.4):
    """Apply power-norm + colormap to a 2D density → (H, W, 3) uint8."""
    normed = np.clip((density / vmax) ** gamma, 0, 1)
    # Transpose + flip for image coords (y-up → row-0-is-top)
    img = normed.T[::-1]
    rgba = _CMAP(img)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _stamp_wall(rgb, V):
    mask = V.T[::-1] > 0
    rgb[mask] = _WALL_COLOR
    return rgb


def _upscale(rgb, size):
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    return np.array(
        Image.fromarray(rgb).resize((size, size), Image.LANCZOS),
    )


def _stamp_text(rgb, text, pos=(12, 10)):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    # Shadow for readability
    sx, sy = pos[0] + 1, pos[1] + 1
    draw.text((sx, sy), text, fill=(0, 0, 0), font=_FONT)
    draw.text(pos, text, fill=(255, 255, 255), font=_FONT)
    return np.array(img)


# ── 1. Propagation ──────────────────────────────────────

def render_propagation(x, y, V, densities, dt, save_every, save_path, fps=25):
    vmax = max(d.max() for d in densities)
    size = _OUTPUT_SIZE

    proc = _open_ffmpeg(save_path, size, size, fps)

    for idx, d in enumerate(densities):
        rgb = _density_to_rgb(d, vmax)
        rgb = _stamp_wall(rgb, V)
        rgb = _upscale(rgb, size)
        t = idx * save_every * dt
        rgb = _stamp_text(rgb, f"t = {t:.2f}")
        proc.stdin.write(rgb.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"Salvo: {save_path}")


# ── 2. Collapse ─────────────────────────────────────────
#
# Strategy: accumulate particle detections into a counts buffer,
# then render with the SAME inferno colormap + power norm as propagation.
# As N → ∞, the image converges to the exact propagation frame.

def render_collapse(
    x, y, V, psi_final,
    n_particles=2_000_000,
    particles_per_frame=5000,
    save_path=None,
    fps=25,
):
    Nx, Ny = psi_final.shape
    size = _OUTPUT_SIZE

    # Sample all particle positions up front
    prob = np.abs(psi_final) ** 2
    prob_flat = prob.ravel()
    prob_flat /= prob_flat.sum()

    print(f"  Amostrando {n_particles:,} partículas...", end=" ", flush=True)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(prob_flat), size=n_particles, p=prob_flat)
    ix, iy = np.unravel_index(indices, prob.shape)
    print("ok")

    n_frames = n_particles // particles_per_frame
    counts = np.zeros((Nx, Ny), dtype=np.float64)

    proc = _open_ffmpeg(save_path, size, size, fps)

    for frame in range(n_frames):
        start = frame * particles_per_frame
        end = start + particles_per_frame

        # Accumulate detections
        np.add.at(counts, (ix[start:end], iy[start:end]), 1)

        # Render with same colormap as propagation
        vmax = counts.max() if counts.max() > 0 else 1.0
        rgb = _density_to_rgb(counts, vmax, gamma=0.4)
        rgb = _stamp_wall(rgb, V)
        rgb = _upscale(rgb, size)
        rgb = _stamp_text(rgb, f"N = {end:,}")
        proc.stdin.write(rgb.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"Salvo: {save_path}")
