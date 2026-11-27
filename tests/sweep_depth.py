"""Depth sweep 0->8 — the headline ordering probe (CLAUDE.md §0.4 / handoff #7).

The project's falsifiable hypothesis: as the gain hook moves shallow->deep, structure should
progress geometry (shallow, high-res cross-attn) -> forms -> figures (deep, semantic bottleneck).
This is the CLEAN single-frame version (no feedback confound): fix everything, sweep ONLY depth
across all 9 flat hookable blocks, at a fixed high gain. Run on BOTH flat-gray (pure generation,
integrity-clean) and the scene (does depth organize scene content into figures?).

Prediction to test:
  - shallow (idx 0-1): periodic lattice (the established ~24px form constant)
  - deep   (idx 3-4 bottleneck): IF the hypothesis holds, figure/blob-scale semantic structure;
    prior probes suggest it may instead be flat/deepfry slop -> that's a clean PARTIAL FALSIFICATION
    (geometry emerges, figures do NOT emerge from depth alone). Report honestly; no face detector
    installed, so judge figure-ness visually + with the periodicity drop.

Run from project root:  .venv/bin/python tests/sweep_depth.py
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


OUT_DIR = "output/sweeps/depth"
DEPTHS = list(range(9))  # 0-3 down, 4 mid bottleneck, 5-8 up
GAIN = 5.0


def main() -> None:
    cfg = RunConfig()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)
    caption = engine.caption_scene(cfg.scene_path)
    print(f"  caption: {caption!r}")

    flat = Image.new("RGB", (cfg.width, cfg.height), (127, 127, 127))
    scene = Image.open(cfg.scene_path).convert("RGB").resize((cfg.width, cfg.height))

    def render(src, cap, prior, depth, tag):
        k = Knobs(scene_precision=0.85, prior_precision=prior, gain=GAIN, depth=depth, recurrence=0.5)
        t0 = time.time()
        frame = engine.render_frame(
            pipe, src, src, cap, k, cfg.seed,
            steps=cfg.denoise_steps, logpolar=False, soft_clip=True,
        )
        frame.save(os.path.join(OUT_DIR, f"{tag}.png"))
        print(f"  {tag}  {time.time()-t0:5.1f}s")

    for d in DEPTHS:
        render(flat, "", 1.0, d, f"flat_d{d}")   # pure generation, unconditional
    for d in DEPTHS:
        render(scene, caption, 7.5, d, f"scene_d{d}")  # scene + caption, moderate CFG

    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
