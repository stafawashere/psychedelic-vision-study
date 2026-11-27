"""Validate the C-3 frequency-decoupled guidance split (tensor only, no model download).

C-3 splits the CFG guidance direction into a low and a high spatial band and weights them
separately. It is a SIXTH DECLARED MECHANIC: any result it touches is claimed as "with a declared
frequency-decoupled guidance", never as the five knobs alone. Off by default in every config.

Six assertions, from docs/development-plan.md section 15.7:

  1. exact reconstruction  low + high rebuilds the direction, and high_ratio 1.0 is the identity
  2. off-path identity     sigma 0.0 returns the input untouched, and no config carries a split
  3. filter calibration    monotone rolloff, sigma doubling halves the cutoff, both axes agree
  4. determinism           two calls on identical inputs are byte-equal
  5. band routing          high_ratio 0.0 removes the high band and leaves the low band alone
  6. dtype and device      output is fp32 on the input's device

Run: .venv/bin/python tests/test_guidance_split.py
Breaks that must be observed red: development-plan section 15.7, B1 to B7. A gate whose breaks were
not all seen red does not count.
"""
from __future__ import annotations
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import RunConfig
from src.render import engine


SHAPE = (1, 4, 64, 64)
SEED = 0
LATENT_CELL_PX = 8  # a 64x64 latent grid renders a 512x512 image

# configs allowed to carry a non-zero guidance_sigma. mirrors LEGACY_SCENE_CONFIGS in
# scripts/check_integrity.py: what stops a split leaking into a deliverable config silently.
SPLIT_GUIDANCE_CONFIGS = frozenset({
    "c3_split_probe.yaml",
})

RECONSTRUCTION_TOLERANCE = 1e-5
DOUBLING_TOLERANCE = 0.10
ORIENTATION_TOLERANCE = 0.05
HIGH_BAND_LEAK_MAX = 0.05
# section 15.7 originally asked the low band to survive high_ratio 0.0 within 1 percent. No gaussian
# can do that at any sigma: energy transfer is exp(-4 pi^2 sigma^2 / w^2), which is 0.25 at the
# half-power wavelength by construction, and 99 percent needs w >= 2 pi sigma / 0.1 = 62.8 cells,
# the whole 64-cell grid. Mahfuj replaced it 2026-08-29 with the pair below, which is satisfiable and
# still reddens both failures the assertion exists for (B4 ratio on the wrong band, B7 split
# discarded). The high-band allowance is unchanged.
LOW_BAND_KEEP_MIN = 0.5
BAND_SELECTIVITY_MIN = 10.0

CALIBRATION_SIGMAS = (0.5, 1.0, 2.0)
PROBE_SIGMA = 1.0  # design choice, declared here and recorded before any render


def _sinusoid(wavelength_cells: float, vertical: bool) -> torch.Tensor:
    grid = torch.arange(SHAPE[-1], dtype=torch.float32)
    wave = torch.cos(2.0 * np.pi * grid / wavelength_cells)
    field = wave.reshape(-1, 1).expand(SHAPE[-2], SHAPE[-1]) if vertical else \
        wave.reshape(1, -1).expand(SHAPE[-2], SHAPE[-1])
    return field.reshape(1, 1, SHAPE[-2], SHAPE[-1]).expand(SHAPE).contiguous()


def retained_fraction(wavelength_cells: float, sigma: float, vertical: bool = False) -> float:
    """Amplitude the lowpass keeps at one wavelength, scored as an RMS ratio.

    Scored on RMS rather than on peak sample value. A sinusoid whose period does not divide the grid
    has its continuous peak between samples, so a peak ratio reads a sampling artifact that is not a
    property of the filter: it made the rolloff non-monotone and moved the measured cutoff.
    """
    field = _sinusoid(wavelength_cells, vertical)
    blurred = engine._lowpass(field, sigma)

    # scored on the interior, away from the reflect-padded edge. a sinusoid whose period does not
    # divide the 64-cell grid meets its own reflection at a kink, and the blur across that kink
    # moves the RMS by about 0.1 percent, which showed up as a non-monotone passband. that is the
    # filter's boundary handling, not its rolloff, and the rolloff is what this measures.
    margin = 2 * int(np.ceil(3.0 * sigma))
    interior = (slice(None), slice(None), slice(margin, -margin), slice(margin, -margin))

    return float(blurred[interior].std() / field[interior].std())


def half_power_wavelength(sigma: float, vertical: bool = False) -> float:
    """Wavelength in latent cells at which the lowpass retains half the amplitude.

    Measured, never derived from a formula: section 9 item 17 and playbook 21 both forbid letting an
    analytic cutoff stand in for the filter the code actually applies. Found by bisection rather than
    by interpolating the printed 32-point curve, whose 2-cell spacing is coarser than the cutoff it
    is trying to locate. The curve is still printed as the record; only the readout is finer.
    """
    low, high = 2.0, 64.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if retained_fraction(middle, sigma, vertical) < 0.5:
            low = middle
        else:
            high = middle

    return 0.5 * (low + high)


def check_reconstruction(failures: list[str]) -> None:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    direction = torch.randn(SHAPE, generator=generator, dtype=torch.float32)

    worst_split = 0.0
    worst_identity = 0.0
    for sigma in CALIBRATION_SIGMAS:
        low = engine._lowpass(direction, sigma)
        high = direction - low
        worst_split = max(worst_split, float((low + high - direction).abs().max()))
        identity = engine._split_guidance(direction, sigma, 1.0)
        worst_identity = max(worst_identity, float((identity - direction).abs().max()))

    print(f"  1 reconstruction     worst |low + high - d| = {worst_split:.3e}   "
          f"worst |identity - d| = {worst_identity:.3e}   tolerance {RECONSTRUCTION_TOLERANCE:.0e}")
    if worst_split > RECONSTRUCTION_TOLERANCE:
        failures.append(f"assertion 1: the split does not reconstruct, worst error {worst_split:.3e}")
    if worst_identity > RECONSTRUCTION_TOLERANCE:
        failures.append(f"assertion 1: high_ratio 1.0 is not the identity, worst error {worst_identity:.3e}")


def check_off_path_identity(failures: list[str]) -> None:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    direction = torch.randn(SHAPE, generator=generator, dtype=torch.float32)

    calls = 0
    original = engine._lowpass

    def counting_lowpass(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    engine._lowpass = counting_lowpass
    try:
        returned = engine._split_guidance(direction, 0.0, 0.5)
    finally:
        engine._lowpass = original

    untouched = returned is direction
    print(f"  2 off-path identity  sigma 0.0 returns the input object = {untouched}   "
          f"_lowpass calls = {calls}")
    if not untouched or calls != 0:
        failures.append("assertion 2: sigma 0.0 did not select the existing single-scalar path")

    # the dataclass default and the from_yaml fallback are written independently in this codebase,
    # so a break that moves one leaves the other untouched. both are checked or neither is guarded.
    defaults = RunConfig()
    print(f"    RunConfig defaults sigma={defaults.guidance_sigma} high_ratio={defaults.guidance_high_ratio}")
    if defaults.guidance_sigma != 0.0 or defaults.guidance_high_ratio != 1.0:
        failures.append("assertion 2: RunConfig defaults are not the off path")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write("dose: {}\nknobs: {}\n")
        path = handle.name
    try:
        parsed = RunConfig.from_yaml(path)
    finally:
        os.unlink(path)

    print(f"    field-free yaml    sigma={parsed.guidance_sigma} high_ratio={parsed.guidance_high_ratio}")
    if parsed.guidance_sigma != 0.0 or parsed.guidance_high_ratio != 1.0:
        failures.append("assertion 2: a yaml with no split fields did not parse to the off path")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    leaked = []
    for name in sorted(os.listdir(os.path.join(root, "configs"))):
        if not name.endswith(".yaml") or name in SPLIT_GUIDANCE_CONFIGS:
            continue
        config = RunConfig.from_yaml(os.path.join(root, "configs", name))
        if config.guidance_sigma != 0.0:
            leaked.append(f"{name} (sigma {config.guidance_sigma})")

    print(f"    config scan        {len(leaked)} configs outside the allowlist carry a split")
    if leaked:
        failures.append(f"assertion 2: split leaked into configs outside the allowlist: {leaked}")


def check_calibration(failures: list[str]) -> None:
    wavelengths = np.linspace(2.0, 64.0, 32)
    retained = np.array([retained_fraction(float(w), PROBE_SIGMA) for w in wavelengths])
    monotone = bool(np.all(np.diff(retained) >= -1e-6))

    single = half_power_wavelength(PROBE_SIGMA)
    doubled = half_power_wavelength(2.0 * PROBE_SIGMA)
    ratio = doubled / single

    vertical = half_power_wavelength(PROBE_SIGMA, vertical=True)
    orientation_gap = abs(vertical - single) / single

    print(f"  3 calibration        sigma {PROBE_SIGMA}: half-power at {single:.3f} cells "
          f"= {single * LATENT_CELL_PX:.1f} px image")
    print(f"    sigma doubling     {single:.3f} -> {doubled:.3f} cells, ratio {ratio:.4f}   "
          f"expect 2.0 +/- {DOUBLING_TOLERANCE:.0%}")
    print(f"    orientation        horizontal {single:.3f} vs vertical {vertical:.3f} cells, "
          f"gap {orientation_gap:.2%}   tolerance {ORIENTATION_TOLERANCE:.0%}")
    print(f"    rolloff monotone in 1/wavelength = {monotone}")

    if not monotone:
        failures.append("assertion 3: the retained-amplitude curve is not monotone")
    if abs(ratio - 2.0) > 2.0 * DOUBLING_TOLERANCE:
        failures.append(f"assertion 3: doubling sigma moved the cutoff by {ratio:.4f}, not 2.0")
    if orientation_gap > ORIENTATION_TOLERANCE:
        failures.append(f"assertion 3: horizontal and vertical cutoffs disagree by {orientation_gap:.2%}")


def check_determinism(failures: list[str]) -> None:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    direction = torch.randn(SHAPE, generator=generator, dtype=torch.float32)

    first = engine._split_guidance(direction, PROBE_SIGMA, 0.5)
    second = engine._split_guidance(direction, PROBE_SIGMA, 0.5)
    identical = torch.equal(first, second)

    print(f"  4 determinism        two calls byte-equal = {identical}")
    if not identical:
        failures.append("assertion 4: the split is not deterministic")


def _band_energy(field: torch.Tensor, cutoff_cells: float, above: bool) -> float:
    spectrum = torch.fft.fftshift(torch.fft.fft2(field.to(torch.float32)), dim=(-2, -1))
    size = field.shape[-1]
    axis = torch.fft.fftshift(torch.fft.fftfreq(size)) * size
    radius = torch.sqrt(axis.reshape(-1, 1) ** 2 + axis.reshape(1, -1) ** 2)

    # a spatial frequency of k cycles across the grid is a wavelength of size / k cells
    wavelength = torch.where(radius > 0, size / radius.clamp(min=1e-6), torch.full_like(radius, 1e6))
    mask = wavelength < cutoff_cells if above else wavelength >= cutoff_cells

    return float((spectrum.abs() ** 2 * mask).sum())


def check_band_routing(failures: list[str]) -> None:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    direction = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
    cutoff = half_power_wavelength(PROBE_SIGMA)

    attenuated = engine._split_guidance(direction, PROBE_SIGMA, 0.0)
    high_keep = _band_energy(attenuated, cutoff, above=True) / _band_energy(direction, cutoff, above=True)
    low_keep = _band_energy(attenuated, cutoff, above=False) / _band_energy(direction, cutoff, above=False)
    selectivity = low_keep / max(high_keep, 1e-12)

    print(f"  5 band routing       at high_ratio 0.0, high-band energy kept = {high_keep:.4%}   "
          f"max {HIGH_BAND_LEAK_MAX:.0%}")
    print(f"    low band kept      {low_keep:.4f}   min {LOW_BAND_KEEP_MIN}")
    print(f"    selectivity        low / high = {selectivity:.2f}   min {BAND_SELECTIVITY_MIN}")

    if high_keep >= HIGH_BAND_LEAK_MAX:
        failures.append(f"assertion 5: high_ratio 0.0 left {high_keep:.2%} of the high band, "
                        f"at or above {HIGH_BAND_LEAK_MAX:.0%}")
    if low_keep < LOW_BAND_KEEP_MIN:
        failures.append(f"assertion 5: high_ratio 0.0 kept only {low_keep:.4f} of the low band, "
                        f"below {LOW_BAND_KEEP_MIN}")
    if selectivity < BAND_SELECTIVITY_MIN:
        failures.append(f"assertion 5: low-over-high selectivity {selectivity:.2f} is below "
                        f"{BAND_SELECTIVITY_MIN}, so the split is not routing by band")


def check_dtype_and_device(failures: list[str]) -> None:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    direction = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
    result = engine._split_guidance(direction, PROBE_SIGMA, 0.5)

    dtype_ok = result.dtype == torch.float32
    device_ok = result.device == direction.device

    print(f"  6 dtype and device   fp32 = {dtype_ok}   same device = {device_ok}   ({result.dtype})")
    if not dtype_ok:
        failures.append(f"assertion 6: the split returned {result.dtype}, not fp32 (section 3 rule)")
    if not device_ok:
        failures.append("assertion 6: the split moved the tensor off the input's device")


def main() -> None:
    print(f"C-3 guidance split gate   shape {SHAPE}, seed {SEED}, probe sigma {PROBE_SIGMA}")

    cutoff = half_power_wavelength(PROBE_SIGMA)
    print(f"\nmeasured half-power wavelength at sigma {PROBE_SIGMA} = {cutoff:.3f} latent cells "
          f"= {cutoff * LATENT_CELL_PX:.1f} px image")
    print("record this before any render; it is measured, never derived from a cutoff formula\n")

    failures: list[str] = []
    check_reconstruction(failures)
    check_off_path_identity(failures)
    check_calibration(failures)
    check_determinism(failures)
    check_band_routing(failures)
    check_dtype_and_device(failures)

    print()
    if failures:
        for message in failures:
            print(f"FAIL: {message}")
        raise SystemExit(1)

    print("all guidance-split checks passed")


if __name__ == "__main__":
    main()
