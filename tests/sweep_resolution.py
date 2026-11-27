"""Resolution probe (backlog #1) — does a bigger latent give finer/sharper geometry?

The crispness ceiling is the VAE upsampling idx0's 32x32 latent 8x to 512px. A bigger render
resolution = bigger latent = idx0 grid is finer, which MIGHT yield finer cells / sharper walls.
Flat-gray + unconditional (no caption) = the integrity-clean pure-generation signal, no foliage
confound. idx0, soft-clip. Vary ONLY resolution (and a gain pair to bracket).

Watch: wavelength may leave the 18-32px lattice band as cells get finer (then judge visually, the
scorer reads 0). Run from project root:  .venv/bin/python tests/sweep_resolution.py
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


OUT_DIR = "output/sweeps/resolution"
DEPTH = 0
RESOLUTIONS = [512, 640, 768]
GAINS = [3.0, 5.0]


def main() -> None:
    cfg = RunConfig()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)

    for res in RESOLUTIONS:
        flat = Image.new("RGB", (res, res), (127, 127, 127))
        for g in GAINS:
            # unconditional (prior=1.0 -> no CFG, caption ignored); recurrence=0.5 -> ds 0.7;
            # prev=flat so it's denoise-depth only, no feedback. ONLY resolution + gain vary.
            k = Knobs(scene_precision=0.85, prior_precision=1.0, gain=g, depth=DEPTH, recurrence=0.5)
            tag = f"res{res}_g{g:g}"
            t0 = time.time()
            frame = engine.render_frame(
                pipe, flat, flat, "", k, cfg.seed,
                steps=cfg.denoise_steps, logpolar=False, soft_clip=True,
            )
            frame.save(os.path.join(OUT_DIR, f"{tag}.png"))
            print(f"  {tag}  {time.time()-t0:5.1f}s")

    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
