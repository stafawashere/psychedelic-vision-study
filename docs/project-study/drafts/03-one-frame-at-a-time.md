# What does the next frame get to see?

## Question

The forest road gets two chances to show up in each frame. One is the actual photo I feed in. The other is whatever survived in the frame the model just made. What happens if I turn the first one down and let the second one feed back? I wanted to see whether the scene could give way to organized geometry and then come back as the dose wore off.

## Five knobs

Follow one frame. The VAE squeezes the photo down into a latent. I blend in scene evidence at a weight called `scene_precision`, and `recurrence` controls how much of the previous frame carries over. Then the UNet works on the noisy blend. The caption gets a vote through `prior_precision`. Inside one block, a hook catches its output and multiplies the forward signal. How hard I multiply = `gain`. Where I clip the hook = `depth`. In this config it stays clipped to the first down block. The skip copy still takes its normal route across.

Decode that and I've got the next frame. Do it 40 times with one rising and falling dose curve. Denoise strength also follows the curve, so a high-dose frame starts further back in noise.

## Rules

Same frozen Stable Diffusion 1.5, same neutral forest-road caption throughout. The photo weight stays above zero. I don't draw geometry or add a warp in arm A. Arm B is a separate run with a declared spatial map. The scores and their cutoffs were set before the runs they judge.

## Scoreboard

- Come-up missed. Seed 0 still had a fine-detail match of 0.672 at frame 05. Five frames later it was 0.066. I wanted detail to hold through frame 11.
- Geometry and symmetry missed. The peak repetition score averaged 1.80 across three seeds. The photo scored 0.44 and my cutoff was above 5. The peak barely matched itself under reflection or a 60-degree turn.
- Breakthrough motion missed. Hold the knobs at the top and frame-to-frame change dies down at all three seeds.
- Return passed on all three. Mean pixel error against frame 00 was 0.0277 and mean fine-detail match was 0.846. My limits were at most 0.05 and at least 0.8 respectively. The integrity checks passed too.

## Finding

Seed 0 makes the rise and fall easy to follow. The repetition score goes 0.42 at the start, 0.99 at peak, 0.43 at the end. For a while the model is filling the image with texture of its own. Once the dose comes down that extra repetition goes away and the fine detail of the road lines up again.

## Limits

That last part depends on what I mean by "same." Frame 00 is already a render. I compared frame 39 to that render, and I kept feeding the original photo in all along. When I jittered the input, the one return I measured scored 0.686 on fine detail. I kept it separate because jitter alone can use up the pixel-error allowance. The three-seed pass belongs to the steady input. These scores tell me what the pixels did, as far as this setup can measure.

[Full report and jitter measurements](../experiments/results.md)
