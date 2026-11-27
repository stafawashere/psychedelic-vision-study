"""Stitch the saved frames into a video (mp4) and a gif."""
from __future__ import annotations
import os

import imageio.v2 as imageio


def _frame_paths(frame_dir: str) -> list[str]:
    names = [f for f in os.listdir(frame_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    names.sort()  # zero-padded indices sort into trip order
    return [os.path.join(frame_dir, n) for n in names]


def _load(frame_dir: str, hold_ends: int) -> list:
    # read frames once; hold_ends repeats first/last so the scene + return read on loop
    paths = _frame_paths(frame_dir)
    if not paths:
        raise SystemExit(f"no frames in {frame_dir!r}")
    frames = [imageio.imread(p) for p in paths]
    if hold_ends > 0:
        frames = [frames[0]] * hold_ends + frames + [frames[-1]] * hold_ends
    return frames


def _write(frames: list, out_path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith(".gif"):
        imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    print(f"wrote {out_path}  ({len(frames)} frames @ {fps}fps)")


def frames_to_video(frame_dir: str, out_path: str, fps: int = 4, hold_ends: int = 0) -> None:
    # single output, format chosen by extension (.mp4 -> ffmpeg, .gif -> pillow)
    _write(_load(frame_dir, hold_ends), out_path, fps)


def frames_to_media(frame_dir: str, out_base: str, fps: int = 4, hold_ends: int = 0) -> None:
    # write BOTH <out_base>.mp4 and <out_base>.gif from a single read of the frames
    frames = _load(frame_dir, hold_ends)
    _write(frames, out_base + ".mp4", fps)
    _write(frames, out_base + ".gif", fps)


if __name__ == "__main__":
    import sys

    frame_dir = sys.argv[1] if len(sys.argv) > 1 else "output/latest/frames"
    out_base = sys.argv[2] if len(sys.argv) > 2 else "output/latest/trip"
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    frames_to_media(frame_dir, out_base, fps=fps, hold_ends=2)
