"""Executable form of the CLAUDE.md section 0 integrity rules.

Checks every config in configs/ plus the knob and dose directions, so a rule
break fails verification instead of silently invalidating renders.
"""
from __future__ import annotations
import ast
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.core.config import KnobShaping
from src.core.dose import dose
from src.core.config import DoseConfig
from src.core.knobs import knobs_from_dose, prior_precision_from_dose


ENGINE_PATH = os.path.join("src", "render", "engine.py")
MAX_BLOCK_INDEX = 8  # flat hookable_blocks: 4 down + mid + 4 up (CLAUDE.md section 8)
RETURN_FRACTION = 0.6  # scene_precision must recover this much of its drop by the last frame

# owner decision 2026-08-29, reaffirmed: resources/scene.jpg is THE reference scene for the whole
# development plan, so a scene swap is a separate experiment with its own baseline rather than a
# variation inside one. these configs predate the decision and are frozen scene-robustness runs.
# scene.png is the canonical path as of 2026-08-29; it is byte-identical to scene.jpg resized
# to 512x512 through run.py's own load (max abs diff 0), so the switch changed no render.
REFERENCE_SCENE = os.path.join("resources", "scene.png")
LEGACY_SCENE_CONFIGS = frozenset({
   "closed_arc_beach.yaml",
   "closed_arc_beach_setcap.yaml",
   "closed_arc_forest_sun.yaml",
   "closed_arc_rock.yaml",
   "closed_arc_rock_ctrl.yaml",
   "foliage.yaml",
})

# E-8's noise substrate and C-3's guidance split are DECLARED MECHANICS OUTSIDE THE 5 KNOBS, so a
# config carrying either must be named here and claimed as such. This is what stops one leaking
# into a deliverable config silently, the way fall: 6.0 did. Adding a name is Mahfuj's call.
NOISE_DRIFT_CONFIGS = frozenset({
   "hold_drift_on.yaml",
   "deliverable_drift_on.yaml",
   "e8_hold_on_s0.yaml",
   "e8_hold_on_s1.yaml",
   "e8_hold_on_s2.yaml",
})
SPLIT_GUIDANCE_CONFIGS = frozenset({
   "c3_split_probe.yaml",
})


def biased_words() -> frozenset[str]:
   tree = ast.parse(open(ENGINE_PATH).read())

   for node in ast.walk(tree):
      is_assign = isinstance(node, ast.Assign)
      if not is_assign:
         continue
      targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
      if "BIASED_WORDS" in targets:
         return frozenset(ast.literal_eval(node.value.args[0]))

   raise SystemExit("BIASED_WORDS not found in engine.py, so the bias filter is gone")


def load_configs() -> list[tuple[str, dict]]:
   paths = sorted(glob.glob(os.path.join("configs", "*.yaml")))
   return [(os.path.basename(p), yaml.safe_load(open(p)) or {}) for p in paths]


def check_configs(failures: list[str]) -> None:
   words = biased_words()

   for name, data in load_configs():
      knobs = data.get("knobs", {}) or {}

      scene_min = knobs.get("scene_min", KnobShaping.scene_min)
      if scene_min <= 0:
         failures.append(f"{name}: scene_min={scene_min} must stay > 0 so the scene can return (rule 3)")

      scheduler = data.get("scheduler", "ddim")
      if scheduler != "ddim":
         failures.append(f"{name}: scheduler={scheduler!r}, only deterministic ddim is allowed")

      dtype = data.get("dtype", "float32")
      if dtype != "float32" and "bf16" not in name:
         failures.append(f"{name}: dtype={dtype!r}, fp32 is the rule; bf16 only as a named opt-in config")

      scene_path = data.get("scene_path", REFERENCE_SCENE)
      is_legacy = name in LEGACY_SCENE_CONFIGS
      uses_reference = os.path.normpath(scene_path) == REFERENCE_SCENE
      if not uses_reference and not is_legacy:
         failures.append(
            f"{name}: scene_path={scene_path!r} is not {REFERENCE_SCENE}, the reference scene for the "
            "whole development plan (owner decision). A scene swap is a separate experiment with its "
            "own baseline; it needs Mahfuj's decision plus a LEGACY_SCENE_CONFIGS entry, not a config"
         )

      anchors = data.get("noise_anchors", 0)
      if anchors and name not in NOISE_DRIFT_CONFIGS:
         failures.append(
            f"{name}: noise_anchors={anchors} but the config is not in NOISE_DRIFT_CONFIGS. The E-8 "
            "noise substrate is a declared mechanic outside the 5 knobs and must be named before a "
            "config may carry it"
         )

      sigma = data.get("guidance_sigma", 0.0)
      if sigma and name not in SPLIT_GUIDANCE_CONFIGS:
         failures.append(
            f"{name}: guidance_sigma={sigma} but the config is not in SPLIT_GUIDANCE_CONFIGS. The C-3 "
            "guidance split is a sixth declared mechanic and must be named before a config may "
            "carry it"
         )

      caption = data.get("caption")
      if caption:
         tokens = {t.lower() for t in re.split(r"\W+", caption) if t}
         hits = sorted(tokens & words)
         if hits:
            failures.append(f"{name}: caption contains biased vocabulary {hits} (rule 2)")


def check_directions(failures: list[str]) -> None:
   for label, shaping, curve in shaping_variants():
      check_one_shaping(label, shaping, curve, failures)


def shaping_variants() -> list[tuple[str, KnobShaping, DoseConfig]]:
   """Defaults plus every config, because a per-config shaping can break a direction the defaults keep."""
   variants = [("defaults", KnobShaping(), DoseConfig())]

   for name, data in load_configs():
      knobs = {k: v for k, v in (data.get("knobs") or {}).items() if k in KnobShaping.__dataclass_fields__}
      dose_cfg = {k: v for k, v in (data.get("dose") or {}).items() if k in DoseConfig.__dataclass_fields__}
      variants.append((name, KnobShaping(**knobs), DoseConfig(**dose_cfg)))

   return variants


PRIOR_SAMPLES = 200  # rest/peak/end alone cannot see the interior of the curve


def check_prior_shape(label: str, shaping: KnobShaping, failures: list[str]) -> None:
   """prior_precision is a pure function of D, so its whole shape is checkable, not just 3 points.

   The monotone mapping must fall everywhere. The ALBUS non-monotone mapping (E-6) must rise below
   d_turn, fall above it, and still land on prior_w_min at full dose. Sampling only rest/peak/end
   passes a curve that does anything at all in between.
   """
   is_non_monotone = shaping.prior_w_peak > 0.0 and 0.0 < shaping.d_turn < 1.0
   samples = [i / (PRIOR_SAMPLES - 1) for i in range(PRIOR_SAMPLES)]
   weights = [prior_precision_from_dose(d, shaping) for d in samples]

   if not is_non_monotone:
      if shaping.prior_w_peak > 0.0 or shaping.d_turn > 0.0:
         failures.append(
            f"{label}: prior_w_peak {shaping.prior_w_peak} / d_turn {shaping.d_turn} is a partial "
            "ALBUS shape; both must be set (d_turn strictly between 0 and 1) or neither"
         )

      falls_everywhere = all(b <= a for a, b in zip(weights, weights[1:]))
      if not falls_everywhere:
         failures.append(f"{label}: monotone prior_precision does not fall across the whole dose range")

      return

   if shaping.prior_w_peak <= shaping.prior_w_base:
      failures.append(
         f"{label}: prior_w_peak {shaping.prior_w_peak} is not above prior_w_base "
         f"{shaping.prior_w_base}, so there is no ALBUS rise to test"
      )
   if shaping.prior_w_peak <= shaping.prior_w_min:
      failures.append(f"{label}: prior_w_peak {shaping.prior_w_peak} is not above prior_w_min {shaping.prior_w_min}")

   rising = [(a, b) for d, a, b in zip(samples[1:], weights, weights[1:]) if d <= shaping.d_turn]
   falling = [(a, b) for d, a, b in zip(samples[1:], weights, weights[1:]) if d > shaping.d_turn]

   if not all(b >= a for a, b in rising):
      failures.append(f"{label}: prior_precision does not rise monotonically below d_turn {shaping.d_turn}")
   if not all(b <= a for a, b in falling):
      failures.append(f"{label}: prior_precision does not fall monotonically above d_turn {shaping.d_turn}")

   at_full_dose = prior_precision_from_dose(1.0, shaping)
   if abs(at_full_dose - shaping.prior_w_min) > 1e-9:
      failures.append(
         f"{label}: prior_precision is {at_full_dose:.4g} at full dose, not prior_w_min "
         f"{shaping.prior_w_min}, so REBUS's collapse never completes"
      )


def check_one_shaping(label: str, shaping: KnobShaping, curve: DoseConfig, failures: list[str]) -> None:
   d_start = dose(0.0, curve)
   d_peak = dose(curve.peak_min, curve)
   d_end = dose(curve.total_min, curve)

   rest = knobs_from_dose(d_start, shaping)
   peak = knobs_from_dose(d_peak, shaping)
   end = knobs_from_dose(d_end, shaping)

   if not (0 < d_start < d_peak and d_end < d_peak):
      failures.append(f"{label}: dose curve is not rise-peak-fall with nonzero endpoints")

   if not peak.scene_precision < rest.scene_precision:
      failures.append(f"{label}: scene_precision does not FALL with dose (must stop trusting the eyes at peak)")
   if peak.scene_precision <= 0:
      failures.append(f"{label}: scene_precision hits <= 0 at peak, so the trip cannot return")
   if not peak.prior_precision < rest.prior_precision:
      failures.append(f"{label}: prior_precision does not fall with dose (REBUS direction)")
   check_prior_shape(label, shaping, failures)
   if not peak.gain > rest.gain:
      failures.append(f"{label}: gain does not rise with dose")
   if not peak.recurrence >= rest.recurrence:
      failures.append(f"{label}: recurrence does not rise with dose")

   # the section 0 deliverable is a breakthrough AND a return; a shaping edit can leave the
   # trip stranded at peak trust while every direction check above still passes
   span = rest.scene_precision - peak.scene_precision
   if span > 0:
      recovered = (end.scene_precision - peak.scene_precision) / span
      if recovered < RETURN_FRACTION:
         failures.append(
            f"{label}: scene_precision recovers only {recovered:.0%} of its drop by t={curve.total_min:g}min "
            f"(need >= {RETURN_FRACTION:.0%}), so the trip cannot come back to the source scene (section 0)"
         )

   for moment, knobs in (("rest", rest), ("peak", peak), ("end", end)):
      if not 0 <= knobs.depth <= MAX_BLOCK_INDEX:
         failures.append(
            f"{label}: depth={knobs.depth} at {moment} is outside the flat hookable_blocks range "
            f"0..{MAX_BLOCK_INDEX}, so attach would fail after the model load"
         )


def main() -> int:
   failures: list[str] = []
   check_configs(failures)
   check_directions(failures)

   if failures:
      print("integrity check FAILED:")
      for f in failures:
         print(f"  - {f}")
      return 1

   print("integrity check passed (configs + knob directions)")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
