"""P0 gate G0. Runs every metric in src/measure/metrics.py against its positive and negative
controls and prints the rows, because a metric reported without its controls is not evidence.

    .venv/bin/python tests/probe_p0_controls.py [scene.png] [other.jpg]

Backs docs/development-plan.md section 2. No model load, no torch, no renders.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.measure.metrics import (CLOSURE_CONTROL_MAX, CLOSURE_DETAIL_MIN, CLOSURE_RMSE_MAX,
                                 CONTROL_SIZE, CONTROL_WAVELENGTH, DYNAMICS_CHURN_MAX,
                                 DYNAMICS_CONTINUITY_MIN, arc_closure, detail_preservation,
                                 form_constant_controls, form_constant_score, hold_dynamics)

BLUR_SIGMA = 2.0
CONTRAST_FACTOR = 1.4
LATTICE_WEIGHT = 0.3
# docs/development-plan.md section 11 item 1 pre-registers the blurred-scene control below this
BLUR_CONTROL_MAX = 0.2
# definitional, not tuned: rings ARE the pure-radial planform, so the majority of their band power
# has to land on the radial axis, and a plane-wave lattice is stationary so it belongs on neither
AXIS_MAJORITY = 0.5


def _as_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))


def _load(path: str) -> Image.Image:
    return Image.open(path).convert("RGB").resize((CONTROL_SIZE, CONTROL_SIZE))


def _dynamics_controls() -> list[tuple[str, list[Image.Image], str]]:
    rng = np.random.default_rng(0)
    base = form_constant_controls()[0][1]
    grain = rng.standard_normal((CONTROL_SIZE, CONTROL_SIZE)) * 0.05

    # converging: the perturbation on a fixed base shrinks each frame, which is what a frozen-noise
    # constant-knob loop does when x_(t+1) = F(x_t) settles
    converging = [_as_image(base + grain * (0.6 ** i)) for i in range(8)]
    # travelling: the same pattern displaced by a constant step, the signature the metric must find
    travelling = [_as_image(np.roll(base, 3 * i, axis=1)) for i in range(8)]
    # churn: unrelated fields, the 56.20 pole where continuity is gone
    churn = [_as_image(np.roll(base, int(rng.integers(0, CONTROL_SIZE)), axis=0)
                       + rng.standard_normal((CONTROL_SIZE, CONTROL_SIZE)) * 0.6) for _ in range(8)]

    return [("CONTROL converging", converging, "decaying / standing"),
            ("CONTROL travelling", travelling, "sustained / travelling"),
            ("CONTROL churn", churn, "churn")]


RECOVERY_LADDER = (16, 24, 32)   # px; spans the 18-32 band and brackets it on both sides
RECOVERY_TOLERANCE = 0.04        # measured/true must sit within this of 1.0; the radial
                                 # binning alone costs 1.7% at 24px, and a one-bin
                                 # off-by-one costs 3 to 6%, so 10% could not see it


def _hex_lattice(period_px: float, size: int = 512) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:size, 0:size].astype(np.float64)
    field = sum(np.cos(2 * np.pi * (grid_x * np.cos(a) + grid_y * np.sin(a)) / period_px)
                for a in (0.0, np.pi / 3, 2 * np.pi / 3))
    return (field - field.min()) / (field.max() - field.min())


def main() -> None:
    scene_path = sys.argv[1] if len(sys.argv) > 1 else "resources/scene.png"
    other_path = sys.argv[2] if len(sys.argv) > 2 else "resources/scene_beach.jpg"

    scene = _load(scene_path)
    other = _load(other_path)
    blurred = scene.filter(ImageFilter.GaussianBlur(BLUR_SIGMA))
    contrasted = ImageEnhance.Contrast(scene).enhance(CONTRAST_FACTOR)

    # the come-up adds geometry on top of the scene, so M-2 has to stay flat when structure is
    # ADDED and only move when detail is LOST. a band-energy-ratio formulation reads 0.498 here
    lattice = form_constant_controls()[0][1]
    grey = np.asarray(scene.convert("L")).astype(np.float64) / 255.0
    overlaid = _as_image((1.0 - LATTICE_WEIGHT) * grey + LATTICE_WEIGHT * lattice).convert("RGB")

    failures = []

    print(f"M-1 form_constant_score   scene={scene_path}")
    print(f"{'image':<22}{'wavelen':>10}{'bump':>8}{'excess':>11}{'aniso':>12}"
          f"{'mirLR':>8}{'rot60':>8}{'rot90':>8}{'radial':>8}{'angular':>9}{'lpConc':>9}")

    rows = form_constant_controls() + [("source photo", np.asarray(scene.convert("L"), dtype=np.float64) / 255.0)]
    scored = {}
    for name, img in rows:
        m = form_constant_score(img)
        scored[name] = m
        print(f"{name:<22}{m['wavelength']:>8.1f}px{m['bump']:>8.2f}{m['excess']:>10.3g}x"
              f"{m['aniso']:>12.4g}{m['mirror_lr']:>8.3f}{m['rot60']:>8.3f}{m['rot90']:>8.3f}"
              f"{m['radial_frac']:>8.3f}{m['angular_frac']:>9.3f}{m['lp_concentration']:>9.1f}")

    lattice_bump = scored["CONTROL hex lattice"]["bump"]
    if lattice_bump <= 10:
        failures.append(f"M-1: hex lattice control bump {lattice_bump:.2f} is not above 10")

    spiral_conc = scored["CONTROL log spiral"]["lp_concentration"]
    photo_conc = scored["source photo"]["lp_concentration"]
    if spiral_conc <= photo_conc:
        failures.append(f"M-1: log spiral lp_concentration {spiral_conc:.1f} does not beat the photo's {photo_conc:.1f}")

    rings = scored["CONTROL rings"]
    if rings["radial_frac"] <= AXIS_MAJORITY:
        failures.append(f"M-1: rings radial_frac {rings['radial_frac']:.3f} is not a majority "
                        f"of band power — the split is not reading the radial axis")

    lattice = scored["CONTROL hex lattice"]
    if max(lattice["radial_frac"], lattice["angular_frac"]) > AXIS_MAJORITY:
        failures.append(f"M-1: the plane-wave lattice was attributed to an axis "
                        f"({lattice['radial_frac']:.3f} radial, {lattice['angular_frac']:.3f} angular)")

    # M-1's wavelength NUMBER was never checked, only the lattice's bump. E-3's whole result is a
    # wavelength regression, so the metric has to be shown to recover a wavelength it is given.
    print(f"\nM-1 wavelength recovery   (hex lattice built at a known period)")
    print(f"{'true':>8}{'measured':>10}{'ratio':>8}{'bump':>8}")
    for true_px in RECOVERY_LADDER:
        m = form_constant_score(_hex_lattice(true_px))
        ratio = m["wavelength"] / true_px
        print(f"{true_px:>8}{m['wavelength']:>9.1f}px{ratio:>8.2f}{m['bump']:>8.2f}")
        if abs(ratio - 1.0) > RECOVERY_TOLERANCE:
            failures.append(f"M-1: wavelength recovery at {true_px}px read {m['wavelength']:.1f}px "
                            f"(ratio {ratio:.2f}), outside +/-{RECOVERY_TOLERANCE:.0%}")

    print(f"\nM-2 detail_preservation   reference={scene_path}")
    print(f"{'subject':<34}{'M-2':>8}")
    detail_rows = [
        ("POSITIVE scene against itself", scene),
        ("POSITIVE contrast inverted", ImageOps.invert(scene)),
        (f"POSITIVE contrast x{CONTRAST_FACTOR}", contrasted),
        (f"POSITIVE {CONTROL_WAVELENGTH}px lattice added at {LATTICE_WEIGHT:g}", overlaid),
        (f"NEGATIVE gaussian blur sigma {BLUR_SIGMA:g}", blurred),
        ("NEGATIVE other photo", other),
    ]
    detail = {}
    for name, img in detail_rows:
        detail[name] = detail_preservation(img, scene)
        print(f"{name:<34}{detail[name]:>8.3f}")

    self_score = detail["POSITIVE scene against itself"]
    blur_score = detail[f"NEGATIVE gaussian blur sigma {BLUR_SIGMA:g}"]
    contrast_score = detail[f"POSITIVE contrast x{CONTRAST_FACTOR}"]
    other_score = detail["NEGATIVE other photo"]

    if abs(self_score - 1.0) > 1e-6:
        failures.append(f"M-2: scene against itself is {self_score:.4f}, not 1")
    if blur_score >= BLUR_CONTROL_MAX:
        failures.append(f"M-2: blur control {blur_score:.3f} is not below {BLUR_CONTROL_MAX}")
    if contrast_score < CLOSURE_DETAIL_MIN:
        failures.append(f"M-2: a pure contrast change dropped the score to {contrast_score:.3f}")

    inverted_score = detail["POSITIVE contrast inverted"]
    if inverted_score < CLOSURE_DETAIL_MIN:
        failures.append(f"M-2: contrast inversion scored {inverted_score:.3f} — it is correlating "
                        f"intensity, not orientation")
    overlaid_score = detail[f"POSITIVE {CONTROL_WAVELENGTH}px lattice added at {LATTICE_WEIGHT:g}"]
    if overlaid_score < CLOSURE_DETAIL_MIN:
        failures.append(f"M-2: added geometry dropped the score to {overlaid_score:.3f} — "
                        f"it is reading structure, not agreement")
    if abs(other_score) > 0.2:
        failures.append(f"M-2: different photo scores {other_score:.3f}, should be near zero")

    print("\nM-3 hold_dynamics")
    print(f"{'segment':<22}{'change':>9}{'first':>8}{'last':>8}{'slope':>8}{'regime':>11}"
          f"{'peakCorr':>10}{'verdict':>12}{'expected':>22}")
    dynamics = {}
    for name, frames, expected in _dynamics_controls():
        m = hold_dynamics(frames)
        dynamics[name] = m
        print(f"{name:<22}{m['mean_change']:>9.2f}{m['first_change']:>8.2f}{m['last_change']:>8.2f}"
              f"{m['slope']:>8.2f}{m['regime']:>11}{m['peak_corr']:>10.3f}{m['verdict']:>12}{expected:>22}")

    if dynamics["CONTROL converging"]["regime"] != "decaying":
        failures.append("M-3: the converging control did not read as decaying")
    if dynamics["CONTROL travelling"]["verdict"] != "travelling":
        failures.append(f"M-3: the travelling control read as {dynamics['CONTROL travelling']['verdict']}")
    if dynamics["CONTROL churn"]["verdict"] != "churn":
        failures.append(f"M-3: the churn control read as {dynamics['CONTROL churn']['verdict']}")
    if dynamics["CONTROL travelling"]["peak_corr"] < DYNAMICS_CONTINUITY_MIN:
        failures.append("M-3: the travelling control lost lag-1 continuity")

    print(f"\nM-4 arc_closure   reference={scene_path}")
    print(f"{'frame':<34}{'rmse':>8}{'M-2':>8}{'closed':>8}")
    closure_rows = [
        ("POSITIVE scene against itself", scene),
        (f"MARGINAL gaussian blur sigma {BLUR_SIGMA:g}", blurred),
        ("NEGATIVE different photo", other),
    ]
    closure = {}
    for name, img in closure_rows:
        m = arc_closure(img, scene)
        closure[name] = m
        print(f"{name:<34}{m['rmse']:>8.4f}{m['detail']:>8.3f}{str(m['closed']):>8}")

    print(f"\nthresholds, pre-registered: rmse <= {CLOSURE_RMSE_MAX}, M-2 >= {CLOSURE_DETAIL_MIN}, "
          f"different-photo control < {CLOSURE_CONTROL_MAX}, churn floor {DYNAMICS_CHURN_MAX}")

    if not closure["POSITIVE scene against itself"]["closed"]:
        failures.append("M-4: the scene does not close against itself")
    if closure["NEGATIVE different photo"]["closed"]:
        failures.append("M-4: a different photo scored as closed")
    if closure["NEGATIVE different photo"]["detail"] >= CLOSURE_CONTROL_MAX:
        failures.append(f"M-4: different-photo M-2 {closure['NEGATIVE different photo']['detail']:.3f} "
                        f"is not below {CLOSURE_CONTROL_MAX}")

    print("")
    if failures:
        print("G0: FAIL")
        for line in failures:
            print(f"  {line}")
        raise SystemExit(1)

    print("G0: PASS — all four metrics separated their controls. P1 may start.")


if __name__ == "__main__":
    main()
