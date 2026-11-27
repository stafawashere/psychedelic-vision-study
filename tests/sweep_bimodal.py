"""E-5 — bimodal multi-block gain (idx0 and idx8 high, mid low).

Declared set, fixed before the run (development-plan.md section 4): idx0 alone as the control,
[0,8] at weights 1.0/0.6, [0,8] at 1.0/1.0, [0,4,8] at 1.0/0.3/0.6. Weights map to gain as
g = 1 + gain_slope * D_apex * w, so w=0 is the identity (gain 1.0) and w=1 is the full apex gain.
gain_slope 1.8 and D_apex 1.0 come from configs/crisp_trip.yaml, whose apex is frame 13.

REFUTED-DIRECTION RE-PROPOSAL. Multi-block gain is on the playbook's refuted list (item 5) and
tests/sweep_multiblock.py collapsed [0,8] to a near-flat image at gains 3.0/2.0. The new evidence
is that the 5-HT2A map is bimodal rather than a shallow-to-deep ramp, so the anatomy specifies a
profile with a trough in the middle, which only the [0,4,8] row here expresses. The [0,8] 1.0/0.6
row is close to the earlier 3.0/2.0 attempt and is expected to reproduce its collapse.

Flat gray + empty caption, so nothing in the frame is scene texture and CFG is off (a flat-gray
frame carries no information about prior_precision). Single frame per row, same seed.

Run from project root:  .venv/bin/python tests/sweep_bimodal.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.core.config import RunConfig  # noqa: E402
from src.core.knobs import Knobs  # noqa: E402
from src.measure.metrics import form_constant_controls, form_constant_score  # noqa: E402
from src.render import engine, hooks  # noqa: E402


OUT_DIR = "output/sweeps/bimodal"
BASELINE_DIR = "output/sweeps/multiblock"  # the 2026-06-22 attempt, kept as the degradation baseline
D_APEX = 1.0   # configs/crisp_trip.yaml frame 13
DS_APEX = 0.82
PROFILES = [
    ("CONTROL idx0", [0], [1.0]),
    ("[0,8] 1.0/0.6", [0, 8], [1.0, 0.6]),
    ("[0,8] 1.0/1.0", [0, 8], [1.0, 1.0]),
    ("[0,4,8] 1/.3/.6", [0, 4, 8], [1.0, 0.3, 0.6]),
]


class ClipCounter:
    """wraps hooks._amplify to report how much of the activation the tanh is actually bending."""

    def __init__(self) -> None:
        self.original = hooks._amplify
        self.clipped = 0
        self.total = 0

    def __enter__(self):
        counter = self

        def counting_amplify(activation, gain, logpolar, soft=False):
            scaled = activation * gain
            std = activation.float().std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
            counter.clipped += int((scaled.float().abs() > hooks.CLAMP_STD * std).sum())
            counter.total += int(scaled.numel())
            return counter.original(activation, gain, logpolar, soft)

        hooks._amplify = counting_amplify
        return self

    def __exit__(self, *exc):
        hooks._amplify = self.original

    @property
    def fraction(self) -> float:
        return self.clipped / self.total if self.total else 0.0


def _gains_for(weights: list[float], slope: float) -> list[float]:
    return [1.0 + slope * D_APEX * w for w in weights]


def _row(name: str, gray: np.ndarray, extra: str = "") -> str:
    m = form_constant_score(gray)
    return (f"{name:<18}{m['wavelength']:>8.1f}px{m['bump']:>8.2f}{m['excess']:>11.3g}x"
            f"{m['aniso']:>12.4g}{m['mirror_lr']:>8.3f}{m['rot60']:>8.3f}"
            f"{m['lp_concentration']:>10.1f}{gray.std():>8.4f}  {extra}")


def _gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def main() -> None:
    cfg = RunConfig.from_yaml("configs/crisp_trip.yaml")
    slope = cfg.knobs.gain_slope
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"apex D={D_APEX}  gain_slope={slope}  ds={DS_APEX}  seed={cfg.seed}  arm={cfg.geometry_arm}")
    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)

    flat = Image.new("RGB", (cfg.width, cfg.height), (127, 127, 127))
    rendered = []

    for name, depths, weights in PROFILES:
        gains = _gains_for(weights, slope)
        tag = "b" + "-".join(str(d) for d in depths) + "_w" + "-".join(f"{w:g}" for w in weights)
        print(f"  rendering {name:<18} depths={depths} gains={[round(g, 3) for g in gains]}")

        knobs = Knobs(scene_precision=0.85, prior_precision=1.0, gain=gains, depth=depths, recurrence=0.5)
        with ClipCounter() as counter:
            frame = engine.render_frame(
                pipe, flat, flat, "", knobs, cfg.seed,
                steps=cfg.denoise_steps, logpolar=False, soft_clip=True, denoise_strength=DS_APEX,
            )

        path = os.path.join(OUT_DIR, f"{tag}.png")
        frame.save(path)
        gray = _gray(path)
        nan_free = bool(np.isfinite(gray).all())
        rendered.append((name, gray, f"clip {counter.fraction * 100:5.2f}%  finite {nan_free}"))

    header = (f"{'row':<18}{'wavelen':>10}{'bump':>8}{'excess':>12}{'aniso':>12}"
              f"{'mirLR':>8}{'rot60':>8}{'lpConc':>10}{'std':>8}")

    print("\nM-1 SYNTHETIC CONTROLS (measurement positive control, same code path)")
    print(header)
    for cname, cimg in form_constant_controls():
        print(_row(cname.replace("CONTROL ", "CTRL "), cimg))

    print("\nBASELINE, the 2026-06-22 multi-block attempt that was refuted")
    print(header)
    for fname, label in [("b0.png", "old [0] g3.0"), ("b0-8.png", "old [0,8] g3/2"), ("b0-7-8.png", "old [0,7,8]")]:
        path = os.path.join(BASELINE_DIR, fname)
        if os.path.exists(path):
            print(_row(label, _gray(path)))

    print("\nE-5 DECLARED PROFILE SET, flat gray, apex gain")
    print(header)
    for name, gray, extra in rendered:
        print(_row(name, gray, extra))

    control_bump = form_constant_score(rendered[0][1])["bump"]
    beats = [n for n, g, _ in rendered[1:] if form_constant_score(g)["bump"] > control_bump]
    print(f"\ncontrol idx0 bump {control_bump:.2f}; bimodal rows above it: {beats or 'none'}")
    print("falsification as pre-registered: every bimodal row below the control, or a reproduced")
    print("collapse (image std near the old 0.0066), closes multi-block gain permanently.")
    print(f"frames -> {OUT_DIR}")


if __name__ == "__main__":
    main()
