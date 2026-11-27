"""E3 — multi-block hook (idx0 + full-res up-block idx7/8) for finer cell edges.

idx0 sets the ~24px macro-lattice at the 32x32 latent, then the VAE upsamples 8x and rounds every
edge. idx7 (640x64x64) / idx8 (320x64x64) operate at the FULL 64x64 latent — the finest scale the
VAE preserves — on §8's strong-survival end of the U. Adding a smaller secondary gain there should
sharpen cell walls without moving the macro-wavelength. Built on E2's winner (ds=0.92, CLAMP_STD=6,
idx0 gain 3). Flat-gray + no caption. ABLATE: [0]-only is the control (= E2 ds0.92_g3).

Run from project root:  .venv/bin/python tests/sweep_multiblock.py
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


OUT_DIR = "output/sweeps/multiblock"
DS = 0.92  # E2 winner
# (depths, gains) sets. [0] is the control; idx0 macro gain 3, secondary up-block gain 2.
SETS = [
    ([0], [3.0]),
    ([0, 7], [3.0, 2.0]),
    ([0, 8], [3.0, 2.0]),
    ([0, 7, 8], [3.0, 2.0, 2.0]),
]


def main() -> None:
    cfg = RunConfig()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)

    flat = Image.new("RGB", (cfg.width, cfg.height), (127, 127, 127))

    for depths, gains in SETS:
        tag = "b" + "-".join(str(d) for d in depths)
        print(f"  rendering {tag}  depths={depths} gains={gains}")
        # pass lists through Knobs.depth/gain -> hooks.attach handles multi-block
        knobs = Knobs(scene_precision=0.85, prior_precision=1.0, gain=gains, depth=depths, recurrence=0.5)
        frame = engine.render_frame(
            pipe, flat, flat, "", knobs, cfg.seed,
            steps=cfg.denoise_steps, logpolar=False, soft_clip=True, denoise_strength=DS,
        )
        frame.save(os.path.join(OUT_DIR, f"{tag}.png"))

    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
