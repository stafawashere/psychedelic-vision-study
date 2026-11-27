# Study notebook

My two original handwritten study notes are [VAE and latent space](vae-latent.md) and [UNet anatomy and hooks](unet.md). The companion pages below were drafted retrospectively from the implementation and experiment records with writing assistance. Their order follows the questions the project raises.

## Build one frame

1. [Sensory evidence](project-study/sensory-evidence.md) — what can the photo tell the model that the caption can't?
2. [The five knobs](project-study/five-knobs.md) — where do the numbers actually go?
3. [Dose and time](project-study/dose-and-time.md) — why is frame 13 the peak?
4. [Caption guidance](project-study/caption-and-guidance.md) — how does a sentence move the prediction?
5. [DDIM and denoise strength](project-study/scheduler-and-noise.md) — where do I enter the noise schedule?
6. [Frame feedback](project-study/frame-feedback.md) — which copy of the frame gets fed back?
7. [Gain and saturation](project-study/gain-and-saturation.md) — what survives normalization and clipping?

## Decide what the pixels show

1. [Seeds and controls](project-study/seeds-and-controls.md) — which observations count as repeats?
2. [Spectrum and wavelength](project-study/spectrum-and-wavelength.md) — what if the ruler finds a spacing in noise?
3. [Log-polar coordinates](project-study/logpolar-and-symmetry.md) — what changes when I unwrap a ring?
4. [Detail and closure](project-study/detail-and-closure.md) — are the branches in the same places?
5. [Sensor jitter](project-study/sensor-jitter.md) — how much error can moving the input introduce?
6. [Hold and motion](project-study/hold-and-motion.md) — is the pattern traveling, settling or turning into static?

## Follow the experiments

1. [Depth and figures](project-study/depth-and-figures.md) — does amplifying the gist bring out a figure?
2. [Two guidance peaks](project-study/two-peaks.md) — why would the same setting fire again on the way home?
3. [Spatial kernels](project-study/spatial-kernel.md) — can I even measure the predicted spacing?
4. [Follow an injection](project-study/follow-an-injection.md) — what survives between the first block and the image?
5. [Guidance by frequency](project-study/guidance-by-frequency.md) — does extra high-frequency energy preserve the branches?
6. [Noise anchors](project-study/noise-anchors.md) — why did moving the peak cost me the ending?
7. [The run record](project-study/run-record.md) — which input, caption and frame does this number belong to?
8. [Test the hook](project-study/testing-the-hook.md) — how do I catch a test that checks its own copy of the code?
9. [Frames and playback](project-study/from-frames-to-video.md) — which clock and which file am I measuring?
