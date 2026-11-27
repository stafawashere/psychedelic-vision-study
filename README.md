# Forest-road diffusion study

The [research guide](docs/README.md) connects the [paper](docs/paper/forest-road-preprint.pdf), [study notebook](docs/notes/README.md) and [experiment results](docs/experiments/results.md). The study follows a forest road through 40 renders of a frozen Stable Diffusion model and measures the departure, peak and return across three seeds.

## Inspect an existing run

The results folders hold the original 40 PNG frames and resolved configuration for each of three seeds. In each folder, frame 00 is the first render, 13 is the peak, and 39 is the return. The [metric reference](docs/experiments/measurements.md) defines the structure and detail scores.

| Seed | Run configuration | Archived frames |
|---|---|---|
| 0 | [Deliverable](configs/deliverable_fall3.yaml) | [Seed 0](results/seed-0/frames) |
| 1 | [First repeat](configs/closure_seed1.yaml) | [Seed 1](results/seed-1/frames) |
| 2 | [Second repeat](configs/closure_seed2.yaml) | [Seed 2](results/seed-2/frames) |

## Run locally

Use Python 3.11 and Apple Silicon for the supplied render configurations. Run these commands from the repository root.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
scripts/verify.sh
.venv/bin/python tests/probe_apex_structure.py results/seed-0/frames 0 13 39
```

To render the main arc, run `.venv/bin/python run.py configs/deliverable_fall3.yaml`. The render downloads the model weights on first use and writes to `output/fall3/`. The archived comparison frames stay in `results/`.

## Project layout

- `src/core/` contains the configuration, dose curve and knob definitions.
- `src/render/` contains the denoiser integration and activation hooks.
- `src/measure/` contains image metrics and video assembly.
- `tests/` contains validation checks and experimental probes.
- `configs/` and `resources/` contain the run settings and input scenes.
- `docs/` contains study notes, measurements and the lattice experiment.
