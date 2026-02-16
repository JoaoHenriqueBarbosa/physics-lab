#!/usr/bin/env python3
"""
Experimento 04 — Teste de Bell com OpenGL

Gera bell_test.mp4: 4 painéis sincronizados (1920×1080, 25fps, 60s)
mostrando a violação da desigualdade de Bell via CHSH.

Uso:
    source /home/john/projects/physics-lab/.gpu-env
    python main.py --save
"""

import argparse
import subprocess
import sys
import time

import numpy as np

from bell import Timeline, TOTAL_FRAMES, FPS
from renderer import BellRenderer, W, H


def main():
    parser = argparse.ArgumentParser(description="Bell test visualization")
    parser.add_argument("--save", action="store_true", help="Render to bell_test.mp4")
    parser.add_argument("--frames", type=int, default=TOTAL_FRAMES,
                        help=f"Number of frames (default {TOTAL_FRAMES})")
    parser.add_argument("--preview", type=int, default=None,
                        help="Render a single frame and save as PNG")
    args = parser.parse_args()

    n_frames = args.frames

    print(f"Inicializando renderer OpenGL (EGL)...")
    renderer = BellRenderer()
    timeline = Timeline()
    print(f"  Contexto GL: {renderer.ctx.info['GL_RENDERER']}")
    print(f"  Resolução: {W}×{H}, {FPS}fps, {n_frames} frames ({n_frames/FPS:.1f}s)")

    if args.preview is not None:
        frame = min(args.preview, n_frames - 1)
        print(f"  Preview frame {frame}...")
        for f in range(frame + 1):
            state = timeline.get_state(f)
        img = renderer.render_frame(state)
        out = f"preview_{frame:04d}.png"
        img.save(out)
        print(f"  Salvo: {out}")
        renderer.release()
        return

    if not args.save:
        print("Use --save para gerar o vídeo ou --preview N para um frame.")
        renderer.release()
        return

    out_path = "bell_test.mp4"
    print(f"  Saída: {out_path}")

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         out_path],
        stdin=subprocess.PIPE,
    )

    t0 = time.time()
    for frame in range(n_frames):
        state = timeline.get_state(frame)
        img = renderer.render_frame(state)

        proc.stdin.write(img.tobytes())

        if (frame + 1) % 100 == 0 or frame == n_frames - 1:
            elapsed = time.time() - t0
            fps_actual = (frame + 1) / elapsed
            eta = (n_frames - frame - 1) / fps_actual
            chsh = state["chsh_s"]
            print(f"  Frame {frame+1:4d}/{n_frames} | "
                  f"{fps_actual:.1f} fps | "
                  f"ETA {eta:.0f}s | "
                  f"N={state['n_measurements']:5d} | "
                  f"S={chsh:+.3f}")

    proc.stdin.close()
    proc.wait()

    elapsed = time.time() - t0
    print(f"\nPronto! {out_path} ({elapsed:.1f}s, {n_frames/elapsed:.1f} fps)")
    renderer.release()


if __name__ == "__main__":
    main()
