from __future__ import annotations
import os
import sys
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from src.core.config import RunConfig  # noqa: E402
from src.core.dose import map_dose  # noqa: E402
from src.core.knobs import knobs_from_dose, lerp  # noqa: E402
from src.render import engine  # noqa: E402
from src.measure.make_video import frames_to_media  # noqa: E402


STABILIZE_CONTRAST = True

# sensor jitter on x_scene, off unless SCENE_JITTER=1 in the environment, so the deliverable
# stream stays bit-identical. amplitudes declared before the run and not tuned afterwards.
JITTER_TRANSLATE_PX = 1.0
JITTER_NOISE_SIGMA = 2.0 / 255.0


def _jitter_scene(scene: Image.Image, seed: int, frame_index: int) -> Image.Image:
    rng = np.random.default_rng([seed, frame_index])
    dx, dy = rng.uniform(-JITTER_TRANSLATE_PX, JITTER_TRANSLATE_PX, size=2)

    # edge-replicated pad so the subpixel shift does not introduce a black border,
    # which would be a framing artifact rather than sensor jitter
    margin = 2
    padded = Image.fromarray(np.pad(np.asarray(scene), ((margin, margin), (margin, margin), (0, 0)), mode="edge"))
    shifted = padded.transform(padded.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR)
    cropped = shifted.crop((margin, margin, margin + scene.width, margin + scene.height))

    values = np.asarray(cropped, dtype=np.float32) / 255.0
    values = values + rng.normal(0.0, JITTER_NOISE_SIGMA, values.shape)

    return Image.fromarray(np.clip(values * 255.0, 0, 255).astype(np.uint8))


def _match_contrast(frame: Image.Image, ref: Image.Image) -> Image.Image:
    f = np.asarray(frame, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    out = np.empty_like(f)
    for c in range(3):
        fc = f[..., c]
        rc = r[..., c]
        out[..., c] = (fc - fc.mean()) * (rc.std() / (fc.std() + 1e-5)) + rc.mean()
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def archive_existing(out_dir: str) -> None:
    if not os.path.isdir(out_dir) or not os.listdir(out_dir):
        return

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest = os.path.join("output", "past", stamp)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(out_dir, dest)


def main(config_path: str | None = None) -> None:
    cfg = RunConfig.from_yaml(config_path) if config_path else RunConfig()
    arm_is_logpolar = cfg.geometry_arm == "B_logpolar_bias"  # arm B warps; arm A never does
    print(f"seed: {cfg.seed}  arm: {cfg.geometry_arm}")

    schedule = map_dose(cfg.dose)
    sampled_peak = max(d for _, d in schedule)
    peak_frame = max(range(len(schedule)), key=lambda i: schedule[i][1])
    if abs(sampled_peak - cfg.dose.dmax) > 1e-9:
        raise SystemExit(
            f"sampled peak {sampled_peak:.6f} < dmax {cfg.dose.dmax:.6f} "
            f"(nearest frame {peak_frame} at t={schedule[peak_frame][0]:.3f}min). "
            "The schedule never lands on the apex; adjust n_frames or hold_min"
        )

    out_dir = cfg.out_dir
    archive_existing(out_dir)
    frame_dir = os.path.join(out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    if not os.path.exists(cfg.scene_path):
        raise SystemExit(f"no scene image at {cfg.scene_path!r}, drop one there first")

    print("loading pipeline")
    pipe = engine.load_pipeline(cfg.model_id, cfg.device, cfg.dtype, cfg.scheduler)

    caption = cfg.caption if cfg.caption is not None else engine.caption_scene(cfg.scene_path)
    print(f"  caption: {caption!r}")

    # snapshot the resolved config, caption included, because it is the run's pre-registration
    cfg.caption = caption
    cfg.to_yaml(os.path.join(out_dir, "config.yaml"))

    scene = Image.open(cfg.scene_path).convert("RGB").resize((cfg.width, cfg.height))
    scene.save(os.path.join(out_dir, "scene.png"))

    pad = len(str(len(schedule) - 1))
    decouple_ds = cfg.knobs.ds_max > 0.0

    jitter_scene = os.environ.get("SCENE_JITTER") == "1"
    if jitter_scene:
        print(f"scene jitter ON: +/-{JITTER_TRANSLATE_PX}px bilinear, gaussian sigma {JITTER_NOISE_SIGMA:.5f}")

    # E-8 noise drift. off unless the config asks for it, and printed when on so a drift run is
    # never mistaken for a plain one in a log. declared substrate, outside the 5 knobs.
    drift_on = cfg.noise_anchors > 0
    noise_shape = (1, 4, cfg.height // 8, cfg.width // 8)
    if drift_on:
        print(f"noise drift ON: anchors={cfg.noise_anchors} phase={cfg.noise_phase:.4f} (declared substrate, outside the 5 knobs)")

    # C-3 guidance split. printed when on so a split run is never mistaken for a plain one in a log.
    split_on = cfg.guidance_sigma > 0.0
    if split_on:
        print(f"guidance split ON: sigma={cfg.guidance_sigma:.4f} high_ratio={cfg.guidance_high_ratio:.3f} (declared mechanic, sixth)")

    prev = scene
    for i, (t_min, d) in enumerate(schedule):
        knobs = knobs_from_dose(d, cfg.knobs)
        ds = lerp(cfg.knobs.ds_min, cfg.knobs.ds_max, d) if decouple_ds else None
        print(
            f"  frame {i:>{pad}}  t={t_min:5.1f}min  D={d:.3f}  "
            f"scene={knobs.scene_precision:.3f} prior={knobs.prior_precision:.2f} "
            f"gain={knobs.gain:.3f} depth={knobs.depth} recur={knobs.recurrence:.3f}"
            + (f" ds={ds:.3f}" if ds is not None else "")
        )

        sensed = _jitter_scene(scene, cfg.seed, i) if jitter_scene else scene

        frame = engine.render_frame(
            pipe,
            sensed,
            prev,
            caption,
            knobs,
            cfg.seed,
            steps=cfg.denoise_steps,
            logpolar=arm_is_logpolar,
            soft_clip=True,
            denoise_strength=ds,
            initial_noise=(
                engine._drifting_noise(noise_shape, cfg.noise_anchors, cfg.noise_phase, i, cfg.seed)
                if drift_on else None
            ),
            guidance_sigma=cfg.guidance_sigma,
            guidance_high_ratio=cfg.guidance_high_ratio,
        )

        frame.save(os.path.join(frame_dir, f"{i:0{pad}d}.png"))
        prev = _match_contrast(frame, scene) if STABILIZE_CONTRAST else frame

    frames_to_media(frame_dir, os.path.join(out_dir, "trip"), fps=5, hold_ends=2)
    print(f"done {out_dir}  ({len(schedule)} frames in frames/, trip.mp4 + trip.gif)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
