# What happened to the forest road?

August 29, 2026. I gave frozen Stable Diffusion 1.5 a photo and one caption, "a view of a dirt road in the middle of a forest." The question was whether I could make organized geometry show up just by changing how the model uses the photo, the caption and its own activity. Then I had to get the same road back.

I got a repeating texture at peak and a measurable return at all three seeds. The hard part was the word *organized*. Patches had a spacing I could measure, but flipping or rotating the image didn't bring them into alignment. Here's what passed, what missed, and how I checked.

## Follow seed 0

Arm A gets no spatial warp. I used DDIM, fp32 on MPS, 512 by 512 pixels, and 40 frames over twelve simulated minutes. The main config is `configs/deliverable_fall3.yaml`, with `fall` at 3.0 and `resources/scene.png` as the input. The seed 1 and 2 configs change the seed and output folder. The reporting session also repeated seed 0 and matched its archived frames bit for bit.

RMSE = pixel error against the first render. M-2 = how well the fine edges still line up. Each frame row here is one observation from seed 0.

| frame | t (min) | rmse | M-2 | reading |
|---|---|---|---|---|
| 05 | 1.5 | 0.0485 | 0.672 | come-up |
| 10 | 3.1 | 0.1462 | 0.066 | departing |
| **13** | **4.0** | **0.2766** | 0.022 | **apex, D = 1.000** |
| 20 | 6.2 | 0.1163 | 0.251 | re-entry |
| 30 | 9.2 | 0.0248 | 0.890 | returned |
| **39** | **12.0** | **0.0241** | **0.852** | **closed** |
| CONTROL | | 0.4427 | -0.001 | different photograph, `resources/scene_beach.jpg` |

Frame 13 is where the dose peaks. The repetition score goes from 0.42 at the start to 0.99 there, then back to 0.43 by the end. Look at the control row too. An earlier version had 0.3167 / 0.016 for a different photo. I couldn't recover that pair from any of the 16 candidate images against either starting render. The beach is a named input I can check again, so that's the row I use.

## How the six tests went

**1. Keep detail through the come-up. Missed.** I wanted M-2 above 0.6 through frame 11. Seed 0 still had 0.672 at frame 05. By frame 06 it was 0.509, and by frame 11 it was 0.048. The plan left the reference open. Comparing against frame 00 gets me about five frames. Comparing against the actual photograph, even frame 00 misses. Holding denoise strength flat, reshaping caption guidance and splitting guidance by frequency all failed to recover the required detail. In the last test M-2 stayed between 0.008 and 0.013 over nine frames.

**2. Get a strong periodic peak. Missed.** The three peak bumps were 0.99, 2.69 and 1.71. Mean 1.80, sample SD 0.85. The photo scored 0.44, the no-signal control 0.293, and a known hexagonal lattice 14.54. My target was above 5. Peak spacing was 20.5, 46.5 and 28.4 px, mean 31.8 with sample SD 13.4. Seed 1 alone is enough to break a claim that the spacing stays inside 18 to 32 px. Earlier SDs of 0.695 and 10.9 used the population convention. They're the same three observations.

**3. Get symmetry. Missed in both arms.** At peak in arm A, left-right mirror averaged 0.101 (sample SD 0.059) and 60-degree rotation averaged -0.024 (sample SD 0.021), three seeds each. Known symmetric inputs scored about 0.98 to 0.99. Arm B starts with a different assumption. I give it a log-polar warp to see whether it can bend what the model already produces into something radially organized.

| probe | n | column | arm A | arm B | verdict |
|---|---|---|---|---|---|
| flat field | 6 v 6 | bump | 2.099 sd 0.086 | 1.103 sd 0.450 | separated DOWN |
| flat field | 6 v 6 | lp_concentration | 138.7 sd 33.3 | 49.7 sd 16.0 | separated DOWN |
| flat field | 6 v 6 | radial_frac | 0.040 sd 0.012 | 0.038 sd 0.016 | within floor |
| come-up to apex | 3 v 3 | radial_frac | 0.038 sd 0.006 | 0.023 sd 0.003 | separated DOWN |
| come-up to apex | 3 v 3 | lp_concentration | 76.2 sd 30.3 | 48.2 sd 9.3 | within floor |
| come-up to apex | 3 v 3 | mirror_lr | 0.101, sample SD 0.059 | 0.110 sd 0.070 | within floor |

I tried the scene as well as flat gray because the forest gives the map actual edges to work with. Across the 96 probe frames neither input gave the predicted increase in radial fraction or log-polar concentration. A separate test put a known lattice right into block 0. Its mirror alignment didn't survive to the output. The [activation notes](lattice-probe.md) give the seed rows and the phase-scrambled control.

**4. Keep a pattern moving at peak. Missed in the deliverable config.** Hold the dose still and measure plateau frames 1 through 12. Then try moving just the starting noise. Change here is measured on a 0 to 255 grayscale scale.

| condition | n | mean change | slope | `peak_corr` | verdict |
|---|---|---|---|---|---|
| drift-off | 3 | 3.686, range 2.609 | -0.9972, range 0.5736, 3 of 3 decaying | 0.9973 | standing |
| drift-on | 3 | **31.726**, range 5.561 | +0.4998, range **1.4073**, 2 of 3 sustained | 0.8922 | standing |
| churn control | 1 | | | 0.158 | churn |

Drift got the rate up, and the adjacent patterns still lined up. Two of three seeds sustained the change. The third decayed. In the full drift arc the ending had RMSE 0.1498 and detail 0.028. The peak spacing also hit 3.7 px, the measurement floor. Either miss disqualifies this config. At frame 00 the noise starts on anchor 0. At frame 39 the noise walk is at position 8.153. I gave the model the same road but a different starting point to draw it from.

The old drift-off baseline of 6.15 belongs to `scene.jpg`. Scoring its archived frames again gave 6.152 / -0.3930 / 0.9967 for mean change, slope and peak correlation. The table uses the current PNG. The baseline moved when the input changed.

**5. Return to the first render. Passed on steady input.** Here are all three seeds. The same seed has to pass both limits, pixel error at most 0.05 and detail at least 0.8.

| row | n | rmse | M-2 | closed |
|---|---|---|---|---|
| 5a un-jittered, **gated**, seed 0 | 1 | **0.0241** | **0.852** | **True** |
| 5a un-jittered, **gated**, seed 1 | 1 | **0.0329** | **0.813** | **True** |
| 5a un-jittered, **gated**, seed 2 | 1 | **0.0262** | **0.872** | **True** |
| **5a mean, sd** | **3** | **0.0277, sd 0.0046** | **0.846, sd 0.030** | **3 of 3 close** |
| 5b jittered, reported not gated, seed 0 | 1 | 0.0318 | 0.686 | False |
| CONTROL different photo, `scene_beach.jpg` | 3 | 0.4427 to 0.4531 | -0.001 | False |

Seed 1 makes the margin clear. It cleared the detail test by 0.013. The jittered input gets its own row because jitter alone gave 0.0383 to 0.0517 pixel error at rest, already straddling the cutoff. I ran that condition at seed 0 only.

**6. Check the implementation. Passed.** In the reporting session, `scripts/verify.sh` ended with `VERIFY: PASS`. That covered syntax, banned patterns, config integrity, hooks, noise drift and split guidance. The drift test had six assertions and five deliberately broken cases. Split guidance had six assertions and seven broken cases. Those checks test the machinery I used to make the frames.

## What I can claim

I can push this model into a repeating texture and get close to the first render again. The texture also shows up on flat input, so branches in this photograph aren't required for it. The results are consistent with the model amplifying spacing its own layers already support. I couldn't demonstrate control over that spacing, a center of symmetry, or sustained coherent motion and closure together. The kernel I considered reached a 71.1 px minimum at block 0 against a 64.0 px measurement ceiling. Blocks 7 and 8 had a usable window on paper but went almost blank when hooked. Amplifying the residual path, multiple blocks or a second peak in caption guidance also gave me no config to promote.

Three seeds are three noise realizations of one scene. The photo gets fed back in every frame. And the measured thing here is a render. No one reported an experience in this experiment. Figures never appeared in the earlier depth tests and weren't a requirement of this deliverable.

## Reproduce or inspect

The archived main runs in this repo are in `results/seed-0`, `results/seed-1` and `results/seed-2`. From the repo root:

```bash
.venv/bin/python tests/probe_apex_structure.py results/seed-0/frames 0 13 39
.venv/bin/python tests/probe_p0_controls.py
.venv/bin/python run.py configs/deliverable_fall3.yaml
.venv/bin/python run.py configs/closure_seed1.yaml
.venv/bin/python run.py configs/closure_seed2.yaml
SCENE_JITTER=1 .venv/bin/python run.py configs/deliverable_fall3_jitter.yaml
scripts/verify.sh
```

[Measurement definitions](measurements.md) · [Development plan and decisions](design.md)
