"""E-4 — does gain amplify the residual/skip stream or the forward-path hidden state.

Owner decision A-E4 approved 2026-08-29 ("approve, probe first"). Three modes at the same nominal
apex gain: forward (today's behaviour and the control), residual (amplify output[1:], leave
output[0] alone), both. Flat gray + empty caption, so nothing in the frame is scene texture and CFG
is off. depth 0, which is a down block and therefore the only kind that HAS a residual stream.

HAZARD, declared in advance (CLAUDE.md section 8): a down block's output[0] is also stored as the
last residual, so `both` hits the up path twice and is the NaN-risky mode. It runs last and only if
`residual` came back finite.

Falsification: if the residual frame is indistinguishable from the forward frame at the same gain
(M-1 within control noise), the apical-dendrite distinction has no computational expression in this
architecture and the claim is closed as untestable here.

Run from project root:  .venv/bin/python tests/sweep_gain_mode.py
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
from src.render import engine  # noqa: E402
from tests.sweep_bimodal import ClipCounter  # noqa: E402


OUT_DIR = "output/sweeps/gain_mode"
DEPTH = 0
D_APEX = 1.0    # configs/crisp_trip.yaml frame 13
DS_APEX = 0.82


def _row(name: str, gray: np.ndarray, extra: str = "") -> str:
    m = form_constant_score(gray)
    return (f"{name:<18}{m['wavelength']:>8.1f}px{m['bump']:>8.2f}{m['excess']:>11.3g}x"
            f"{m['aniso']:>12.4g}{m['mirror_lr']:>8.3f}{m['rot60']:>8.3f}"
            f"{m['lp_concentration']:>10.1f}{gray.std():>8.4f}  {extra}")


def main() -> None:
    # optional seed list: the same modes across seeds give M-1's seed-to-seed floor, without which
    # a small bump difference between two modes cannot be read as a difference at all
    seeds = [int(a) for a in sys.argv[1:]]

    cfg = RunConfig.from_yaml("configs/crisp_trip.yaml")
    gain = 1.0 + cfg.knobs.gain_slope * D_APEX
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"depth={DEPTH}  gain={gain}  ds={DS_APEX}  seed={cfg.seed}  arm={cfg.geometry_arm}")
    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)

    flat = Image.new("RGB", (cfg.width, cfg.height), (127, 127, 127))
    knobs = Knobs(scene_precision=0.85, prior_precision=1.0, gain=gain, depth=DEPTH, recurrence=0.5)
    rendered = []

    modes = ["forward", "both"] if seeds else ["forward", "residual", "both"]
    for mode in modes:
        if mode == "both" and not all(row[2] for row in rendered):
            print("  skipping 'both': an earlier mode was not finite, and both hits the up path twice")
            break

        for seed in (seeds or [cfg.seed]):
            print(f"  rendering {mode} seed {seed}")
            with ClipCounter() as counter:
                frame = engine.render_frame(
                    pipe, flat, flat, "", knobs, seed,
                    steps=cfg.denoise_steps, logpolar=False, soft_clip=True,
                    denoise_strength=DS_APEX, gain_mode=mode,
                )

            path = os.path.join(OUT_DIR, f"{mode}_s{seed}.png" if seeds else f"{mode}.png")
            frame.save(path)
            gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
            finite = bool(np.isfinite(gray).all())
            label = f"{mode} s{seed}" if seeds else f"{mode}{' (CONTROL)' if mode == 'forward' else ''}"
            rendered.append((label, gray, finite, counter.fraction))

    header = (f"{'row':<18}{'wavelen':>10}{'bump':>8}{'excess':>12}{'aniso':>12}"
              f"{'mirLR':>8}{'rot60':>8}{'lpConc':>10}{'std':>8}")

    print("\nM-1 SYNTHETIC CONTROLS (measurement positive control, same code path)")
    print(header)
    for cname, cimg in form_constant_controls():
        print(_row(cname.replace("CONTROL ", "CTRL "), cimg))

    print("\nE-4 GAIN MODES, flat gray, apex gain, depth 0")
    print(header)
    for name, gray, finite, clip in rendered:
        print(_row(name, gray, f"clip {clip * 100:5.2f}%  finite {finite}"))

    print()
    if seeds:
        for mode in modes:
            bumps = [form_constant_score(g)["bump"] for n, g, _, _ in rendered if n.startswith(mode)]
            spread = max(bumps) - min(bumps)
            print(f"{mode:<10} bump across seeds {seeds}: "
                  + "  ".join(f"{b:.2f}" for b in bumps) + f"   spread {spread:.2f}")
        print("a mode-to-mode bump difference smaller than the within-mode seed spread is not a result.")
    else:
        control = rendered[0][1]
        for name, gray, _, _ in rendered[1:]:
            pixel_rmse = float(np.sqrt(np.mean((gray - control) ** 2)))
            print(f"{name} against the forward control: pixel rmse {pixel_rmse:.4f}")
    print("falsification: a residual frame indistinguishable from forward at the same gain closes")
    print("the apical-dendrite claim as untestable in this architecture.")
    print(f"frames -> {OUT_DIR}")


if __name__ == "__main__":
    main()
