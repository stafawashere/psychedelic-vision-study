"""Discriminator: does gain GENERATE geometry, or just AMPLIFY scene texture?

The idx-0 gain sweep produced a ~36px honeycomb lattice -- but that scale matches the
forest photo's own texture (~34px), so it might be amplified scene texture, not a de novo
form constant. This batch separates the two:

  A) flat gray + NO caption (unconditional)  -- the purest test. Featureless input, zero
     semantic guidance. A lattice here = the gain mechanism GENERATES geometry from nothing.
  B) flat gray + forest caption (cfg 5)       -- controls for caption-injected texture.
  C) forest + alternate seeds                 -- is the lattice robust or one magic seed?

All on the pinned high-res block idx 0, soft-clip on. Compare against the forest sweep in
output/sweeps/gain_depth_softclip.

Run from project root:  .venv/bin/python tests/sweep_discriminate.py
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


OUT_DIR = "output/sweeps/discriminate"
DEPTH = 0  # the block where the lattice appeared


def knobs(gain, prior):
    # recurrence=0.5 -> denoise_strength 0.7 (matches the forest sweep); prev=scene so it's
    # denoise depth only, not feedback. scene_precision moot (both blend inputs equal).
    return Knobs(scene_precision=0.85, prior_precision=prior, gain=gain, depth=DEPTH, recurrence=0.5)


def main() -> None:
    cfg = RunConfig()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)
    forest_caption = engine.caption_scene(cfg.scene_path)
    print(f"  forest caption: {forest_caption!r}")

    flat = Image.new("RGB", (cfg.width, cfg.height), (127, 127, 127))  # featureless mid-gray
    flat.save(os.path.join(OUT_DIR, "flat_input.png"))
    forest = Image.open(cfg.scene_path).convert("RGB").resize((cfg.width, cfg.height))

    def render(scene, caption, k, seed, tag):
        print(f"  rendering {tag}")
        frame = engine.render_frame(
            pipe, scene, scene, caption, k, seed,
            steps=cfg.denoise_steps, logpolar=False, soft_clip=True,
        )
        frame.save(os.path.join(OUT_DIR, f"{tag}.png"))

    # A) flat + unconditional (prior=1.0 disables CFG, caption ignored): purest generation test
    for g in [1.0, 3.0, 5.0, 8.0]:
        render(flat, "", knobs(g, prior=1.0), cfg.seed, f"flatU_g{g:g}")

    # B) flat + forest caption at cfg 5: controls for caption-injected texture
    for g in [1.0, 8.0]:
        render(flat, forest_caption, knobs(g, prior=5.0), cfg.seed, f"flatC_g{g:g}")

    # C) forest + alternate seeds at the gain where the lattice was strong
    for seed in [1, 2]:
        render(forest, forest_caption, knobs(5.0, prior=5.0), seed, f"forest_g5_s{seed}")

    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
