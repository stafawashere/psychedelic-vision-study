"""measurement probe, NOT a gate. answers one question about a rendered run: is the apex
periodic, and is it symmetric? synthetic controls run alongside so a null result is readable.

    .venv/bin/python tests/probe_apex_structure.py output/past/2026-08-29_05-31-15/frames 0 13 15

backs docs/conclusion.md section 2. the machinery now lives in src/measure/metrics.py as M-1
(`form_constant_score`); this stays a thin caller so the published numbers stay reproducible.
no model load, no torch.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.measure.metrics import form_constant_controls, form_constant_score


def _load_gray(frame_dir: str, index: int) -> np.ndarray:
    for name in sorted(os.listdir(frame_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".png" and stem.isdigit() and int(stem) == index:
            return np.asarray(Image.open(os.path.join(frame_dir, name)).convert("L"), dtype=np.float64) / 255.0

    raise FileNotFoundError(f"no frame {index} in {frame_dir}")


def main() -> None:
    frame_dir = sys.argv[1] if len(sys.argv) > 1 else "output/latest/frames"
    indices = [int(a) for a in sys.argv[2:]] or [0, 13, 15]

    controls = [row for row in form_constant_controls() if row[0] != "CONTROL flat gray"]
    targets = controls + [(f"frame {i:02d}", _load_gray(frame_dir, i)) for i in indices]

    print(f"{'image':<22}{'wavelen':>10}{'bump':>8}{'excess':>11}{'aniso':>12}{'mirLR':>8}{'rot60':>8}{'rot90':>8}")
    for name, img in targets:
        m = form_constant_score(img)
        print(f"{name:<22}{m['wavelength']:>8.1f}px{m['bump']:>8.2f}{m['excess']:>10.3g}x"
              f"{m['aniso']:>12.4g}{m['mirror_lr']:>8.3f}{m['rot60']:>8.3f}{m['rot90']:>8.3f}")

    print("\ncontrols set the dynamic range: a real form constant reads bump > 10 and/or symmetry > 0.9.")
    print("periodicity WITHOUT symmetry means a saturation texture, not a Turing pattern.")


if __name__ == "__main__":
    main()
