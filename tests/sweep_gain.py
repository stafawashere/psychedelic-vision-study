"""Depth-pinned gain sweep — the cleanest chance for arm-A geometric emergence.

Isolate the gain knob. Everything else is held fixed: same seed, same scene anchor,
same neutral caption + CFG, same denoise depth. depth is PINNED to a high-res
cross-attn block (idx 0/1, where lattice-scale structure could form) instead of
diving to the low-res bottleneck the dose schedule sends it to at peak. soft-clip
(tanh) instead of the hard clamp so amplification doesn't just posterize into deepfry.

If structured/periodic texture appears here as gain climbs -> arm A has legs and the
default depth schedule was misrouting the gain. If it's slop at every gain on the
highest-res block -> that's the clean, attributable negative result.

Run from the project root:  .venv/bin/python tests/sweep_gain.py
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


# pinned high-res cross-attn blocks (flat index): 0 = 320x32x32, 1 = 640x16x16
DEPTHS = [0, 1]
# vary ONLY this. 1.0 = control (no hook attaches). soft-clip lets us push hard safely.
GAINS = [1.0, 2.0, 3.0, 5.0, 8.0]

OUT_DIR = "output/sweeps/gain_depth_softclip"


def main() -> None:
    cfg = RunConfig()
    if not os.path.exists(cfg.scene_path):
        raise SystemExit(f"no scene image at {cfg.scene_path!r}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"seed: {cfg.seed}  depths: {DEPTHS}  gains: {GAINS}  soft-clip: ON")

    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)
    caption = engine.caption_scene(cfg.scene_path)
    print(f"  caption: {caption!r}")

    scene = Image.open(cfg.scene_path).convert("RGB").resize((cfg.width, cfg.height))
    scene.save(os.path.join(OUT_DIR, "scene.png"))

    for depth in DEPTHS:
        for gain in GAINS:
            # everything fixed except gain+depth. prev=scene so recurrence sets denoise
            # depth only (no feedback loop): blend of scene+scene is scene regardless.
            # scene_precision moot here (both blend inputs are the scene); recurrence=0.5
            # -> denoise_strength 0.7 gives gain ~21 steps to act while staying scene-anchored.
            knobs = Knobs(
                scene_precision=0.85,
                prior_precision=5.0,
                gain=gain,
                depth=depth,
                recurrence=0.5,
            )
            tag = f"d{depth}_g{gain:g}"
            print(f"  rendering {tag}")
            frame = engine.render_frame(
                pipe, scene, scene, caption, knobs, cfg.seed,
                steps=cfg.denoise_steps, logpolar=False, soft_clip=True,
            )
            frame.save(os.path.join(OUT_DIR, f"{tag}.png"))

    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
