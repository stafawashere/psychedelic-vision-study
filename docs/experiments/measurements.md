# What my four image scores actually measure

I can blur the road until all the little branches disappear and still tell you it's a road. That's why I need different scores for recognizing the scene, keeping the detail, making a pattern and moving that pattern over time. Here's what each one sees and the controls I use to check it.

| | question it answers | signature |
|---|---|---|
| **M-1** | is this a Kluver planform, or merely periodic | `form_constant_score(img, predicted_wavelength=None) -> dict` |
| **M-2** | was scene detail preserved, or destroyed and resynthesised | `detail_preservation(frame, reference) -> float` |
| **M-3** | does a hold segment have dynamics, or only parameter drift | `hold_dynamics(frames) -> dict` |
| **M-4** | did the arc actually come back | `arc_closure(frame, reference) -> dict` |

## M-1. Does one spacing show up more than the others?

`form_constant_score` looks for a peak above the background trend in the frequency spectrum. The **bump** tells me how far the peak sticks up. **Wavelength** tells me where the instrument found it. Then I flip or rotate the picture and measure how well it lines up with itself. A texture can have spacing all over it and barely line up at all.

For rings and spirals I also measure the image in log-polar coordinates. A ring stays the same as I go around it. A spoke stays in the same direction as I go out from the center. Those give me different axes to measure. A spiral concentrates power off those axes. This coordinate change happens while I'm *measuring* the finished frame. Arm B's warp happens inside the model while it's *making* the frame.

## M-2. Are the little edges in the same places?

`detail_preservation(frame, reference)` uses the reference to whiten both images, keeps frequencies above 1/8 Nyquist, then compares the directions of the gradients. Darken an edge or invert its contrast and it can still point along the same line. Blur it away and the agreement drops. Order matters here because the second image supplies the whitening filter.

## M-3. What moves when I stop moving the knobs?

`hold_dynamics` needs at least three frames from a constant-dose hold. Mean change is grayscale RMSE on a 0 to 255 scale. The slope tells me whether that change keeps going or dies down. Then I slide neighboring frames against each other and look at peak correlation and offset. A consistent shift can mean a traveling pattern. A match near zero shift can mean a standing one. Throw random noise into every frame and the change gets big too, but adjacent frames stop lining up.

## M-4. Did I get the same road back?

`arc_closure` measures RGB pixel error on a 0 to 1 scale AND the M-2 detail match. I compare the last render to the first render of the same seed. To close I need RMSE at most 0.05 and detail at least 0.8. The beach gives me a check on how the score behaves when I hand it a different scene.

## Check the instruments

These controls use the 512 by 512 source photo as the reference. That's why the beach row here differs slightly from the beach compared to a *render* in the [deliverable report](results.md).

| metric | positive | negative |
|---|---|---|
| M-1 | hex lattice bump 14.54; rings `radial_frac` 0.985; spiral `lp_concentration` 5788; **wavelength recovery 16/24/32 px reads 16.0/24.4/32.0, ratios 1.00/1.02/1.00** | flat gray all-zero; photo `lp_concentration` 38.5; lattice on neither axis (0.051 / 0.166) |
| M-2 | self 1.000; contrast inverted 1.000; contrast x1.4 0.964; +24px lattice 0.997 | blur sigma 2 at 0.052; different photo 0.001 |
| M-3 | converging slope -0.69 decaying; travelling verdict at `peak_corr` 1.000 | churn verdict at `peak_corr` 0.158 |
| M-4 | scene against itself, rmse 0.0000 / M-2 1.000 / closed | different photo, rmse 0.4535 / M-2 0.001 / not closed |

The other limits in the control suite are 0.3 for different-photo detail, 0.7 for motion continuity, 0.5 for churn and 0.2 for blur. I tightened wavelength recovery from 0.10 to 0.04 because moving the peak by one bin could still pass at 0.10.

## Seven things that tripped me up

1. **Whitening can erase the thing I'm looking for.** Dividing by the measured average at each radius made tiny sampling leftovers huge. The rings control went from 1.000 radial fraction raw to 0.250 with that whitening. I fit a power law over the band instead.
2. **The fine-band cutoff does less than I expected.** At cutoffs of 0, 0.125 and 0.25 Nyquist the blur scores were 0.070, 0.052 and 0.046. I keep the cutoff to separate the fine detail from the roughly 24 px geometry. Add a 24 px lattice and the detail score still reads 0.997.
3. **Direction matters.** Plain intensity scored a contrast inversion at -1.000 and blur at 0.262. The gradient directions correctly kept the inverted edges aligned.
4. **Break the thing a control claims to catch.** One test I thought detected an intensity substitution stayed green. The change it actually caught was replacing the measurement with a band-energy ratio (0.498).
5. **Jitter has a price before the model does anything.** A shift of up to one pixel plus noise at sigma 2/255 gave a rest floor of 0.0383 to 0.0517 RMSE and detail of 0.546 to 0.325. That crosses my 0.05 line. With the perturbation removed I got 0.0077 and 0.982. I report jitter separately.
6. **Same dimensions at render time don't guarantee the same run.** The old 641 by 360 scene was resized to a square. The newer PNG contains that squash already. Even when the resized input pixels agree, the caption and archived hold results can differ. Keep the source filename with the score.
7. **The wavelength readout can find a peak in nothing.** The no-signal control reports 17.8 px with a bump of 0.293. Flat gray can hit the 64.0 px ceiling. On clean lattices the instrument recovers 16, 24 and 32 px within tolerance. On a filtered band over a background it underestimates by 20 to 50 percent. It finds the peak left after subtracting the fitted trend, which can sit well away from the original peak. Across six flat-field seeds, wavelength was 31.0 px with SD 7.5 while bump was 2.10 with SD 0.09. I report the bump first and keep the calibration beside any spacing claim.

## A real run to compare against

In the earlier `scene.jpg` arc, the final frame scored 0.0247 RMSE and 0.843 detail. Frame 19 still scored 0.832 on the old scene-fidelity measure even though detail had fallen to 0.125. That's the failure I wanted M-2 to catch. Bump went 0.44, 1.10, 0.63 and 0.46 at frames 00, 13, 19 and 39. Those numbers belong to that earlier arc.

From the repo root I can run the controls and then measure the archived seed 0 frames:

```bash
.venv/bin/python tests/probe_p0_controls.py
.venv/bin/python tests/probe_apex_structure.py results/seed-0/frames 0 13 39
```

If I add a metric, it goes in `src/measure/metrics.py`. Its negative control goes in `tests/probe_p0_controls.py`. Before trusting the result, I change the implementation in the specific way the control should catch and check that it fails.
