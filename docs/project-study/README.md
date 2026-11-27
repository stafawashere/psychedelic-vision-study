# Forest-road study

Can a frozen image model leave a forest road, make organized geometry and find its way back? On the three main seeds the fine detail of the road came back. A repeating texture showed up at peak, but the patterns missed the symmetry and motion criteria.

| What you want to do | Start here |
|---|---|
| Read the study | [Zenodo preprint](https://doi.org/10.5281/zenodo.24592604) · [Paper (PDF)](paper/forest-road-preprint.pdf) · [Manuscript](paper/manuscript.md) · [Evidence supplement](paper/forest-road-supplement.zip) |
| Check a result | [Frame and seed scores](experiments/results.md) · [Metric definitions](experiments/measurements.md) |
| Understand the experiment | [Design and decisions](experiments/design.md) · [Lattice injection](experiments/lattice-probe.md) |
| Learn how the model works | [Study notebook](#study-notebook) |
| Compare earlier ways of explaining it | [Working drafts](drafts/README.md) |
| Inspect a run or render another | [Setup and archived frames](../../README.md) |

## Study notebook

### Build one frame

1. [Sensory evidence](sensory-evidence.md) — what can the photo tell the model that the caption can't?
2. [VAE and latent space](vae-latent.md) — what does the VAE carry from pixels into the latent grid?
3. [UNet anatomy and hooks](unet.md) — where do features flow, and where does the gain hook intervene?
4. [The five knobs](five-knobs.md) — where do the numbers actually go?
5. [Dose and time](dose-and-time.md) — why is frame 13 the peak?
6. [Caption guidance](caption-and-guidance.md) — how does a sentence move the prediction?
7. [DDIM and denoise strength](scheduler-and-noise.md) — where do I enter the noise schedule?
8. [Frame feedback](frame-feedback.md) — which copy of the frame gets fed back?
9. [Gain and saturation](gain-and-saturation.md) — what survives normalization and clipping?

### Decide what the pixels show

1. [Seeds and controls](seeds-and-controls.md) — which observations count as repeats?
2. [Spectrum and wavelength](spectrum-and-wavelength.md) — what if the ruler finds a spacing in noise?
3. [Log-polar coordinates](logpolar-and-symmetry.md) — what changes when I unwrap a ring?
4. [Detail and closure](detail-and-closure.md) — are the branches in the same places?
5. [Sensor jitter](sensor-jitter.md) — how much error can moving the input introduce?
6. [Hold and motion](hold-and-motion.md) — is the pattern traveling, settling or turning into static?

### Follow the experiments

1. [Depth and figures](depth-and-figures.md) — does amplifying the gist bring out a figure?
2. [Two guidance peaks](two-peaks.md) — why would the same setting fire again on the way home?
3. [Spatial kernels](spatial-kernel.md) — can I even measure the predicted spacing?
4. [Follow an injection](follow-an-injection.md) — what survives between the first block and the image?
5. [Guidance by frequency](guidance-by-frequency.md) — does extra high-frequency energy preserve the branches?
6. [Noise anchors](noise-anchors.md) — why did moving the peak cost me the ending?
7. [The run record](run-record.md) — which input, caption and frame does this number belong to?
8. [Test the hook](testing-the-hook.md) — how do I catch a test that checks its own copy of the code?
9. [Frames and playback](from-frames-to-video.md) — which clock and which file am I measuring?
