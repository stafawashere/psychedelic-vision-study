"""prior_precision (CFG) effect on geometry (backlog #5).

prior_precision is the CFG guidance scale on the neutral caption. Does pushing CFG up/down
sharpen, color, or wash out the lattice? Tested on the SCENE (CFG only bites with a caption;
flat+unconditional ignores it). idx0, soft-clip, gain pinned. Vary ONLY prior_precision.

Run from project root:  .venv/bin/python tests/sweep_prior.py
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PIL import Image  # noqa: E402
from src.core.config import RunConfig  # noqa: E402
from src.core.knobs import Knobs  # noqa: E402
from src.render import engine  # noqa: E402


OUT_DIR = "output/sweeps/prior"
DEPTH = 0
GAIN = 3.0
PRIORS = [1.0, 3.0, 5.0, 7.5, 12.0, 18.0]


def main() -> None:
    cfg = RunConfig()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)
    caption = engine.caption_scene(cfg.scene_path)
    print(f"  caption: {caption!r}")
    scene = Image.open(cfg.scene_path).convert("RGB").resize((cfg.width, cfg.height))

    for prior in PRIORS:
        k = Knobs(scene_precision=0.85, prior_precision=prior, gain=GAIN, depth=DEPTH, recurrence=0.5)
        tag = f"prior{prior:g}"
        t0 = time.time()
        frame = engine.render_frame(
            pipe, scene, scene, caption, k, cfg.seed,
            steps=cfg.denoise_steps, logpolar=False, soft_clip=True,
        )
        frame.save(os.path.join(OUT_DIR, f"{tag}.png"))
        print(f"  {tag}  {time.time()-t0:5.1f}s")

    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
