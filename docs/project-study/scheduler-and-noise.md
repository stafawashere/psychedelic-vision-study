# What DDIM does after the UNet answers

The UNet hands back a noise prediction. I used to picture myself just subtracting that tensor from the image. There's a step missing there. The scheduler knows which timestep I'm at and how much signal and noise belong at the next one. It uses the prediction to work out the latent for that step. The UNet gets another look at that new latent.

## Where do I enter the schedule?

The config says 30 steps. The runner sets up that schedule, then takes a tail of it depending on denoise strength. At the peak ds = 0.82. Round 30 times 0.82 and I get 25 active steps. I add noise at the first timestep of that tail. The peak frame gets a lot of room to be rebuilt.

Now the flat-ds experiment makes sense. Keep ds at 0.45 and I enter closer to the clean end. In that comparison the fine-detail score at frame 10 went from 0.144 to 0.432. But the peak's periodic excess dropped below the photo's own reading. More of the branches survived and the repeating texture got weaker.

> simulated minute picks the knobs. denoising timestep picks the next latent.

[Scheduler calls](../../src/render/engine.py) · [Flat-ds comparison](experiments/design.md)
