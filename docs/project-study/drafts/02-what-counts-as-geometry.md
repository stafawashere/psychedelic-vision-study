# What would actually count as geometry?

## Question

Say the trees disappear and the whole frame fills up with evenly spaced blobs. Did I get geometry? That question gets tricky once I'm looking at the picture, because my eyes are pretty good at joining a few blobs into a shape. So I decided what the image had to do before I saw it.

## Five knobs

I start Stable Diffusion 1.5 from a forest road. The photo is sensory evidence and the caption is a rough stand-in for what the model expects the scene to be. I can turn down the weight of each one (`scene_precision` and `prior_precision`). Inside the UNet I pick a level (`depth`) and turn up its activations (`gain`). Finally I can feed some of what it just made back in (`recurrence`). The idea is that a feature the model picks up can get louder and show up again in the next frame. Depth stayed at block 0 here. The other four knobs follow the same dose curve.

## Rules

Arm A asks what shows up with a neutral caption, some photo input at every frame and zero spatial warp. Arm B gives the model a log-polar map, a piece of wiring that could bend stripes into rings. If that makes rings, I have to say I gave it the map. I set the geometry target in advance and kept every seed.

## Scoreboard

| Arm A test | What I measured | What that buys |
|---|---|---|
| Detail on the way up | Seed 0, M-2 drops under 0.6 by frame 06 | Missed the frame 11 target |
| Repetition at peak | Bump 0.99, 2.69, 1.71 for seeds 0, 1, 2 | All above the photo's 0.44. All below the required 5 |
| Order at peak | Mean mirror score 0.101, 60-degree rotation -0.024, three seeds | Known symmetric patterns score around 0.98 to 0.99 |
| Motion at peak | Change decayed at all three seeds in the hold test | Missed sustained motion |
| Return | Three of three back inside both limits | The forest road comes back |
| Integrity | Full verification passed | Config and hook checks held |

## Finding

I tried giving the first block a perfect hexagonal lattice. Its mirror score going in was 1.000. Coming out of the whole model, the average signed mirror score was 0.120 over eight seeds. That gets at why "just turn it up more" is a poor fix here. I can hand the block the exact structure I wanted and still lose it on the way to pixels. My read is that the rest of the UNet builds a fresh image from those features. Even arm B failed to raise the measures of radial order on either gray input or the road.

## Limits

I can call the arm A peak a saturation texture. I don't have the symmetry I'd need to call it a Klüver form constant, like a tunnel or lattice. The measured spacings jump from 20.5 to 46.5 to 28.4 pixels across seeds, so I can't claim I picked a wavelength either. This tells me where structure gets lost in this image model. It doesn't settle how a visual cortex produces it.

[Full report and lattice controls](../experiments/results.md)
