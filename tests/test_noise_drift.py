"""Validate the E-8 noise-drift substrate (no model download, no renders).

E-8 replaces the per-frame `torch.randn` initial latent noise with a slerp walk through K fixed
anchors, so consecutive frames start from *nearby* noise instead of independent noise. It is a
declared deterministic substrate OUTSIDE the 5 knobs and is named as such in every claim it touches.

Six assertions, from docs/e8-implementation-spec.md section 2.3 plus development-plan section 15.6:

  1. unit variance      every frame's noise std within 0.01 of 1.0 (naive lerp reads ~0.71 and fails)
  2. orthogonality      pairwise |pearson| between anchors below 0.02
  3. continuity         mean consecutive-frame correlation within 0.02 of the 0.95 target
  4. determinism        same seed rebuilds byte-identically, and consecutive frames DIFFER
  5. off-path identity  noise_anchors == 0 selects the existing path, with run.py never calling in
  6. pass-through       the tensor handed to render_frame is the one add_noise receives

Run: .venv/bin/python tests/test_noise_drift.py
Breaks that must be observed red: docs/e8-implementation-spec.md section 2.3 and development-plan
section 15.6. A gate whose breaks were not all seen red does not count.
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
ANCHORS = 3
SEED = 0
FRAMES = 12

CONTINUITY_TARGET = 0.95
CONTINUITY_TOLERANCE = 0.02
STD_TOLERANCE = 0.01
ORTHOGONALITY_MAX = 0.02


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    # scored in numpy on the cpu, never as a torch float64 tensor, since float64 is unsupported
    # on MPS and this file must stay runnable wherever the engine runs
    x = a.flatten().cpu().numpy().astype(np.float64)
    y = b.flatten().cpu().numpy().astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    return float(x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def _anchors(shape, anchors: int, seed: int) -> list[torch.Tensor]:
    """Read the anchors back OUT of the engine, never by redrawing them here.

    At phase 1.0 and frame index k the walk sits exactly on anchor k % anchors, because slerp at
    t = 0 returns its first argument. Re-drawing them locally instead would test a copy of the
    engine rather than the engine: break 2 (duplicated anchor) left the orthogonality check green
    and reddened continuity instead, which is how the hole surfaced.
    """
    return [engine._drifting_noise(shape, anchors, 1.0, k, seed) for k in range(anchors)]


def _sequence(shape, anchors: int, phase: float, seed: int, frames: int) -> list[torch.Tensor]:
    return [engine._drifting_noise(shape, anchors, phase, i, seed) for i in range(frames)]


def _mean_consecutive_correlation(sequence: list[torch.Tensor]) -> float:
    pairs = [_pearson(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
    return float(np.mean(pairs))


def calibrate_noise_phase(shape, anchors: int, seed: int, target: float = CONTINUITY_TARGET,
                          frames: int = FRAMES) -> float:
    """Bisect phase for a target mean consecutive-frame correlation. Zero renders.

    Correlation is monotone decreasing in phase over (0, 0.5) for near-orthogonal anchors, which is
    what makes bisection valid here. The result is recorded in the log and written into the drift-on
    configs BEFORE any render, and is never re-tuned after seeing a result.
    """
    low, high = 0.01, 0.5
    phase = 0.5 * (low + high)

    for _ in range(40):
        phase = 0.5 * (low + high)
        correlation = _mean_consecutive_correlation(_sequence(shape, anchors, phase, seed, frames))

        if abs(correlation - target) <= 0.005:
            return phase

        if correlation > target:
            low = phase
        else:
            high = phase

    return phase


class _StubDistribution:
    def __init__(self, latent: torch.Tensor):
        self.latent = latent

    def sample(self, generator=None) -> torch.Tensor:
        return self.latent


class _StubVaeOutput:
    def __init__(self, tensor: torch.Tensor):
        self.latent_dist = _StubDistribution(tensor)
        self.sample = tensor


class _StubVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        class _Config:
            scaling_factor = 0.18215

        self.config = _Config()

    def encode(self, image_t: torch.Tensor) -> _StubVaeOutput:
        batch, _, height, width = image_t.shape
        return _StubVaeOutput(torch.zeros((batch, 4, height // 8, width // 8), dtype=torch.float32))

    def decode(self, latents: torch.Tensor) -> _StubVaeOutput:
        batch, _, height, width = latents.shape
        return _StubVaeOutput(torch.zeros((batch, 3, height * 8, width * 8), dtype=torch.float32))


class _StubUnetOutput:
    def __init__(self, tensor: torch.Tensor):
        self.sample = tensor


class _StubUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, latent, timestep, encoder_hidden_states=None) -> _StubUnetOutput:
        return _StubUnetOutput(torch.zeros_like(latent))


class _StubStepOutput:
    def __init__(self, tensor: torch.Tensor):
        self.prev_sample = tensor


class _StubScheduler:
    def __init__(self):
        self.timesteps = torch.arange(30, 0, -1)
        self.captured_noise: torch.Tensor | None = None

    def set_timesteps(self, steps, device=None) -> None:
        self.timesteps = torch.arange(steps, 0, -1)

    def add_noise(self, latents, noise, timesteps) -> torch.Tensor:
        self.captured_noise = noise.detach().clone()
        return latents + noise

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, noise_prediction, timestep, latent, generator=None) -> _StubStepOutput:
        return _StubStepOutput(latent)


class _StubPipeline:
    def __init__(self):
        self.device = torch.device("cpu")
        self.vae = _StubVae()
        self.unet = _StubUnet()
        self.scheduler = _StubScheduler()

    def encode_prompt(self, caption, device, num, use_guidance, negative):
        embedding = torch.zeros((1, 77, 32), dtype=torch.float32)
        return embedding, embedding


def check_unit_variance(phase: float, failures: list[str]) -> None:
    sequence = _sequence(SHAPE, ANCHORS, phase, SEED, FRAMES)
    deviations = [abs(float(frame.std()) - 1.0) for frame in sequence]
    worst = max(deviations)

    print(f"  1 unit variance      worst |std - 1| = {worst:.5f}   tolerance {STD_TOLERANCE}")
    if worst > STD_TOLERANCE:
        failures.append(f"assertion 1: a frame's noise std is off by {worst:.5f}, above {STD_TOLERANCE}")

    # the negative control lives in the same check: a naive lerp collapses the norm, which is the
    # whole reason slerp is here rather than the one-line interpolation
    fields = _anchors(SHAPE, ANCHORS, SEED)
    lerp_midpoint = 0.5 * fields[0] + 0.5 * fields[1]
    lerp_std = float(lerp_midpoint.std())

    print(f"    negative control   naive-lerp midpoint std = {lerp_std:.4f}   must fail the same check")
    if abs(lerp_std - 1.0) <= STD_TOLERANCE:
        failures.append("assertion 1 control: a naive lerp midpoint passed the unit-variance check, "
                        "so the check cannot discriminate slerp from lerp")


def check_orthogonality(failures: list[str]) -> None:
    fields = _anchors(SHAPE, ANCHORS, SEED)
    pairs = [(i, j) for i in range(ANCHORS) for j in range(i + 1, ANCHORS)]
    worst = max(abs(_pearson(fields[i], fields[j])) for i, j in pairs)

    print(f"  2 orthogonality      worst pairwise |pearson| = {worst:.5f}   tolerance {ORTHOGONALITY_MAX}")
    if worst >= ORTHOGONALITY_MAX:
        failures.append(f"assertion 2: anchors correlate at {worst:.5f}, at or above {ORTHOGONALITY_MAX}")


def check_continuity(phase: float, failures: list[str]) -> None:
    correlation = _mean_consecutive_correlation(_sequence(SHAPE, ANCHORS, phase, SEED, FRAMES))
    offset = abs(correlation - CONTINUITY_TARGET)

    print(f"  3 continuity         mean consecutive corr = {correlation:.4f}   "
          f"target {CONTINUITY_TARGET} +/- {CONTINUITY_TOLERANCE}")
    if offset > CONTINUITY_TOLERANCE:
        failures.append(f"assertion 3: consecutive correlation {correlation:.4f} is {offset:.4f} "
                        f"from the {CONTINUITY_TARGET} target, above {CONTINUITY_TOLERANCE}")


def check_determinism(phase: float, failures: list[str]) -> None:
    first = _sequence(SHAPE, ANCHORS, phase, SEED, FRAMES)
    second = _sequence(SHAPE, ANCHORS, phase, SEED, FRAMES)
    reproducible = all(torch.equal(a, b) for a, b in zip(first, second))

    # non-degeneracy is a separate half so a phase that ignores the frame index has something to
    # redden: without it, every frame being identical would still pass the determinism half
    distinct = all(not torch.equal(first[i], first[i + 1]) for i in range(len(first) - 1))

    print(f"  4 determinism        rebuild identical = {reproducible}   consecutive frames differ = {distinct}")
    if not reproducible:
        failures.append("assertion 4: rebuilding the sequence from the same seed did not reproduce it")
    if not distinct:
        failures.append("assertion 4: consecutive frames are identical, so the walk is degenerate")


def check_off_path_identity(failures: list[str]) -> None:
    defaults = RunConfig()
    print(f"  5 off-path identity  RunConfig defaults: anchors={defaults.noise_anchors} "
          f"phase={defaults.noise_phase}")
    if defaults.noise_anchors != 0 or defaults.noise_phase != 0.0:
        failures.append("assertion 5: RunConfig defaults are not the off path")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write("dose: {}\nknobs: {}\n")
        path = handle.name
    try:
        parsed = RunConfig.from_yaml(path)
    finally:
        os.unlink(path)

    print(f"    field-free yaml    anchors={parsed.noise_anchors} phase={parsed.noise_phase}")
    if parsed.noise_anchors != 0 or parsed.noise_phase != 0.0:
        failures.append("assertion 5: a yaml with no noise fields did not parse to the off path")

    for anchors, expect_calls in ((0, 0), (ANCHORS, 4)):
        calls, passed = _drive_run_py(anchors)
        supplied = sum(1 for value in passed if value is not None)

        print(f"    run.py anchors={anchors}  _drifting_noise calls={calls}  "
              f"initial_noise supplied on {supplied} of {len(passed)} frames")
        if calls != expect_calls:
            failures.append(f"assertion 5: run.py called _drifting_noise {calls} times at "
                            f"anchors={anchors}, expected {expect_calls}")
        if anchors == 0 and supplied != 0:
            failures.append("assertion 5: run.py passed a non-None initial_noise with drift off")
        if anchors > 0 and supplied != len(passed):
            failures.append("assertion 5: run.py passed initial_noise=None on some frame with drift on")


def _drive_run_py(anchors: int) -> tuple[int, list]:
    """Run run.main against stubs, counting _drifting_noise calls and what render_frame received.

    This is the only place the wiring itself is under test. A tensor-only check cannot see run.py
    forgetting to pass the keyword at all.
    """
    import run as run_module

    original_drifting = engine._drifting_noise
    original_render = engine.render_frame
    original_load = engine.load_pipeline
    original_caption = engine.caption_scene
    original_media = run_module.frames_to_media

    calls = 0
    passed: list = []

    def counting_drifting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_drifting(*args, **kwargs)

    def stub_render(*args, **kwargs):
        passed.append(kwargs.get("initial_noise"))
        from PIL import Image
        return Image.new("RGB", (64, 64), "black")

    with tempfile.TemporaryDirectory() as work_dir:
        config = RunConfig(
            height=64,
            width=64,
            scene_path="resources/scene.png",
            out_dir=os.path.join(work_dir, "out"),
            caption="a neutral caption",
            noise_anchors=anchors,
            noise_phase=0.2,
        )
        config.dose.n_frames = 4
        config_path = os.path.join(work_dir, "cfg.yaml")
        config.to_yaml(config_path)

        engine._drifting_noise = counting_drifting
        engine.render_frame = stub_render
        engine.load_pipeline = lambda *a, **k: _StubPipeline()
        engine.caption_scene = lambda *a, **k: "a neutral caption"
        run_module.frames_to_media = lambda *a, **k: None
        try:
            run_module.main(config_path)
        finally:
            engine._drifting_noise = original_drifting
            engine.render_frame = original_render
            engine.load_pipeline = original_load
            engine.caption_scene = original_caption
            run_module.frames_to_media = original_media

    return calls, passed


def check_pass_through(failures: list[str]) -> None:
    """Assertion 6. The five tensor assertions above all stay green if render_frame accepts
    initial_noise and quietly draws its own, which would null the whole mechanism.
    """
    from PIL import Image
    from src.core.knobs import Knobs

    pipe = _StubPipeline()
    scene = Image.new("RGB", (64, 64), "gray")
    knobs = Knobs(scene_precision=0.8, prior_precision=1.0, gain=1.0, depth=0, recurrence=0.0)
    known = torch.arange(1 * 4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8) / 1000.0

    engine.render_frame(pipe, scene, scene, "a neutral caption", knobs, SEED,
                        steps=2, initial_noise=known)
    captured = pipe.scheduler.captured_noise
    matched = captured is not None and torch.equal(captured.cpu(), known)

    print(f"  6 pass-through       add_noise received the passed tensor = {matched}")
    if not matched:
        failures.append("assertion 6: the tensor passed as initial_noise is not the one add_noise "
                        "received, so render_frame is ignoring the keyword")

    pipe.scheduler.captured_noise = None
    mismatched = torch.zeros((1, 4, 16, 16), dtype=torch.float32)
    try:
        engine.render_frame(pipe, scene, scene, "a neutral caption", knobs, SEED,
                            steps=2, initial_noise=mismatched)
        raised = False
    except ValueError:
        raised = True

    print(f"    shape validation   wrong-shape initial_noise raised ValueError = {raised}")
    if not raised:
        failures.append("assertion 6: a wrong-shaped initial_noise was accepted instead of raising")


def main() -> None:
    print("E-8 noise drift gate   K = 3 anchors, seed 0, shape (1, 4, 64, 64), 12 frames")

    phase = calibrate_noise_phase(SHAPE, ANCHORS, SEED)
    correlation = _mean_consecutive_correlation(_sequence(SHAPE, ANCHORS, phase, SEED, FRAMES))
    print(f"\ncalibrated noise_phase = {phase:.6f}   mean consecutive correlation = {correlation:.4f}")
    print("record this value before any render and never re-tune it after seeing a result\n")

    failures: list[str] = []
    check_unit_variance(phase, failures)
    check_orthogonality(failures)
    check_continuity(phase, failures)
    check_determinism(phase, failures)
    check_off_path_identity(failures)
    check_pass_through(failures)

    print()
    if failures:
        for message in failures:
            print(f"FAIL: {message}")
        raise SystemExit(1)

    print("all noise-drift checks passed")


if __name__ == "__main__":
    main()
