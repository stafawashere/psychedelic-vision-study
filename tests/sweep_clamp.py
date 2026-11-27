"""E1 — CLAMP_STD sweep on idx0 (the dominant crispness lever).

§8: only the NON-uniform part of the soft-clip survives GroupNorm. At CLAMP_STD=6 the tanh barely
bends, so most of `gain` is a uniform multiply that cancels -> soft blobs. Tightening CLAMP_STD makes
the tanh transfer switch-like -> sharper cell walls. We sweep it on a FLAT-GRAY + no-caption input
(the integrity-clean GENERATED-geometry signal, free of foliage-texture confound) and score crispness.

Watch: clip% > 3% = regressing toward deepfry; wavelength must stay in the 18-32px lattice band
(else the 'sharper' frame broke the lattice into noise, didn't crisp it).

Run from project root:  .venv/bin/python tests/sweep_clamp.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PIL import Image  # noqa: E402
from src.core.config import RunConfig  # noqa: E402
from src.core.knobs import Knobs  # noqa: E402
from src.render import engine, hooks  # noqa: E402


OUT_DIR = "output/sweeps/clamp"
DEPTH = 0                       # idx1 is dead (periodicity flat); only the high-res block lives here
CLAMPS = [6.0, 3.0, 2.0, 1.5]  # 6 = validated control; tighter = switch-like transfer
GAINS = [3.0, 5.0]             # established periodicity sweet spot


def main() -> None:
    cfg = RunConfig()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)

    flat = Image.new("RGB", (cfg.width, cfg.height), (127, 127, 127))  # featureless, no texture to amplify

    for clamp in CLAMPS:
        hooks.CLAMP_STD = clamp  # _amplify reads this module global at call time
        for gain in GAINS:
            tag = f"c{clamp:g}_g{gain:g}"
            print(f"  rendering {tag}  (CLAMP_STD={clamp})")
            frame = engine.render_frame(
                pipe, flat, flat, "", Knobs(0.85, 1.0, gain, DEPTH, 0.5), cfg.seed,
                steps=cfg.denoise_steps, logpolar=False, soft_clip=True,
            )
            frame.save(os.path.join(OUT_DIR, f"{tag}.png"))

    hooks.CLAMP_STD = 6.0  # restore default
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
