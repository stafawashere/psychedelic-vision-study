"""E2 — denoise-strength sweep on idx0 (let the lattice crystallize more fully).

Higher denoise_strength starts further back in noise and runs more denoising steps over the same
t-range, so the gain-driven lattice can fully form instead of sitting as a soft mosaic on a barely
denoised field. Flat-gray + no-caption input (clean generated-geometry signal). CLAMP_STD stays 6
(E1 showed tighter is strictly worse). denoise_strength passed EXPLICITLY (decoupled from recurrence).

Run from project root:  .venv/bin/python tests/sweep_denoise.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PIL import Image  # noqa: E402
from src.core.config import RunConfig  # noqa: E402
from src.core.knobs import Knobs  # noqa: E402
from src.render import engine  # noqa: E402


OUT_DIR = "output/sweeps/denoise"
DEPTH = 0
GAINS = [3.0, 5.0]
STRENGTHS = [0.7, 0.85, 0.92]  # 0.7 = current baseline; higher = more crystallization


def main() -> None:
    cfg = RunConfig()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)

    flat = Image.new("RGB", (cfg.width, cfg.height), (127, 127, 127))

    for ds in STRENGTHS:
        for gain in GAINS:
            tag = f"ds{ds:g}_g{gain:g}"
            print(f"  rendering {tag}")
            frame = engine.render_frame(
                pipe, flat, flat, "", Knobs(0.85, 1.0, gain, DEPTH, 0.5), cfg.seed,
                steps=cfg.denoise_steps, logpolar=False, soft_clip=True, denoise_strength=ds,
            )
            frame.save(os.path.join(OUT_DIR, f"{tag}.png"))

    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
