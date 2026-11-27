from __future__ import annotations
from dataclasses import dataclass
from .config import KnobShaping


def lerp(low: float, high: float, x: float) -> float:
    return low + (high - low) * x


@dataclass
class Knobs:
    scene_precision: float # trust
    prior_precision: float # weight
    gain: float # amplifier 1 + gain_slope * D
    depth: int # target block/layer
    recurrence: float # bleed between frames


def prior_precision_from_dose(dose: float, cfg: KnobShaping) -> float:
    # REBUS describes the high-agonism regime only; ALBUS holds that lower occupancy can strengthen
    # priors, so the weight may rise through the come-up before collapsing near apex
    is_non_monotone = cfg.prior_w_peak > 0.0 and 0.0 < cfg.d_turn < 1.0
    if not is_non_monotone:
        return lerp(cfg.prior_w_base, cfg.prior_w_min, dose)

    if dose <= cfg.d_turn:
        return lerp(cfg.prior_w_base, cfg.prior_w_peak, dose / cfg.d_turn)

    return lerp(cfg.prior_w_peak, cfg.prior_w_min, (dose - cfg.d_turn) / (1.0 - cfg.d_turn))


def knobs_from_dose(dose: float, cfg: KnobShaping) -> Knobs:
    return Knobs(
        # sensory precision falls from its ceiling at rest toward the floor at peak
        # ("stop trusting the eyes"); floor (scene_min) stays > 0 so the scene can return
        scene_precision=lerp(cfg.scene_max, cfg.scene_min, dose),
        prior_precision=prior_precision_from_dose(dose, cfg),
        gain=1.0 + cfg.gain_slope * dose,
        depth=round(lerp(cfg.depth_shallow, cfg.depth_deep, dose)),
        recurrence=lerp(cfg.recurrence_min, cfg.recurrence_max, dose),
    ) 