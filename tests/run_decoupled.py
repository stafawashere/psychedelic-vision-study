"""Decoupled-ds trip runner (E-DECOUPLE) — scratch, does NOT touch run.py.

Replicates run.py's loop but DECOUPLES denoise_strength (crystallization depth) from recurrence
(feedback blend weight). Both stay functions of the dose D (integrity: no hand-painting):
  - recurrence (blend bleed): lerp(0, RECUR_MAX, D)   -- kept LOW to reduce feedback streak
  - denoise_strength:         lerp(DS_MIN, DS_MAX, D)  -- kept HIGH at peak to crystallize the lattice

ds is an allowed sampling mechanic (handoff §3). render_frame already accepts an explicit
denoise_strength that overrides the recurrence-derived one. Everything else mirrors run.py
(prior floor, gain, depth 0, soft-clip, contrast stabilizer).

Usage:  .venv/bin/python tests/run_decoupled.py [recur_max] [ds_min] [ds_max] [out_tag]
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from src.core.config import RunConfig  # noqa: E402
from src.core.dose import map_dose  # noqa: E402
from src.core.knobs import knobs_from_dose, Knobs  # noqa: E402
from src.render import engine  # noqa: E402

RECUR_MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.18
DS_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
DS_MAX = float(sys.argv[3]) if len(sys.argv) > 3 else 0.82
TAG = sys.argv[4] if len(sys.argv) > 4 else f"r{RECUR_MAX:g}_ds{DS_MIN:g}-{DS_MAX:g}"

N_FRAMES = 18
PRIOR_BASE, PRIOR_MIN = 12.0, 6.0
GAIN_SLOPE = 1.8


def match_contrast(frame, ref):
    f = np.asarray(frame, dtype=np.float32); r = np.asarray(ref, dtype=np.float32)
    out = np.empty_like(f)
    for c in range(3):
        fc, rc = f[..., c], r[..., c]
        out[..., c] = (fc - fc.mean()) * (rc.std() / (fc.std() + 1e-5)) + rc.mean()
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def lerp(a, b, x):
    return a + (b - a) * x


def main():
    cfg = RunConfig()
    cfg.dose.n_frames = N_FRAMES
    cfg.dose.rise = 1.8
    cfg.dose.fall = 6.0
    out_dir = os.path.join("output/exp", "decouple_" + TAG)
    frame_dir = os.path.join(out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)
    caption = engine.caption_scene(cfg.scene_path)
    print(f"  caption: {caption!r}  recur_max={RECUR_MAX} ds=[{DS_MIN},{DS_MAX}]")
    scene = Image.open(cfg.scene_path).convert("RGB").resize((cfg.width, cfg.height))

    schedule = map_dose(cfg.dose)
    pad = len(str(len(schedule) - 1))
    prev = scene
    for i, (t_min, d) in enumerate(schedule):
        # hand-build knobs so recurrence (blend) and ds (crystallization) are independent functions of D
        knobs = Knobs(
            scene_precision=lerp(0.85, 0.40, d),
            prior_precision=lerp(PRIOR_BASE, PRIOR_MIN, d),
            gain=1.0 + GAIN_SLOPE * d,
            depth=0,
            recurrence=lerp(0.0, RECUR_MAX, d),
        )
        ds = lerp(DS_MIN, DS_MAX, d)
        frame = engine.render_frame(
            pipe, scene, prev, caption, knobs, cfg.seed,
            steps=cfg.denoise_steps, logpolar=False, soft_clip=True, denoise_strength=ds,
        )
        frame.save(os.path.join(frame_dir, f"{i:0{pad}d}.png"))
        prev = match_contrast(frame, scene)
        print(f"  frame {i:>{pad}} D={d:.3f} recur={knobs.recurrence:.3f} ds={ds:.3f} gain={knobs.gain:.2f}")

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
