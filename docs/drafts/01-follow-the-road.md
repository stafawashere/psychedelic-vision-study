# Where did the road go?

## Question

Frame 00 is a dirt road with trees on either side. At frame 13 I'd have a hard time pointing out where the road even is. By frame 39 I can find it again. I wanted to know what happens in between. Can an image model get from a photo into the kind of organized geometry people describe on DMT, then find its way back, just from changing how it uses the information it already has?

## Five knobs

Stable Diffusion 1.5 works on a compressed version of the image. Each time it builds a frame I get to change a few things about that process.

- `scene_precision` = how much weight the actual photo gets. Kind of like trusting your eyes.
- `prior_precision` = how hard the caption steers the image toward what the model expects to see.
- `gain` = how loud one block in the UNet gets. I multiply the features it already picked up.
- `depth` = WHICH block gets louder. I kept it at the first down block for this run.
- `recurrence` = how much of the last frame feeds into the next one.

One dose curve drives the changes. It peaks four minutes into a twelve-minute arc. At the peak the photo and caption have less pull, gain is up and more of the previous frame gets through.

## Rules

The model stays frozen. Its whole prompt is "a view of a dirt road in the middle of a forest." Arm A gets no warp and I draw nothing into the frame. I keep some photo signal going in even at peak. Before rendering I set the scores a run needs to pass, including how close it has to get to frame 00 on the way back.

## Scoreboard

Here's seed 0 out of three. RMSE measures pixel difference from the first render, so smaller = closer. M-2 asks whether the fine detail lines up, so higher = closer.

| Frame | RMSE | M-2 | What I get |
|---|---:|---:|---|
| 05 | 0.0485 | 0.672 | Still carrying detail from the road |
| 13 | 0.2766 | 0.022 | Peak dose. Almost none of that detail lines up |
| 39 | 0.0241 | 0.852 | Back inside both return thresholds |
| Beach photo control | 0.4427 | -0.001 | What a genuinely different scene scores |

I needed M-2 above 0.6 through frame 11 for a detail-preserving come-up. It was already 0.509 by frame 06. The road goes soft early. At peak I get repeating texture, but its average bump is 1.80 over three seeds against a target above 5. Mirror and rotation scores stay near zero. Holding the dose high gives a pattern whose motion dies down. Of the six deliverable checks, return and code integrity passed.

## Finding

I can lose the road in a repeating texture and get it back. All three seeds closed. The distinction that matters to me is what "repeating" buys here. You can get roughly the same spacing between patches all over an image and still have no center, no rings, nothing that lines up when you rotate it. That's the kind of repetition I got.

## Limits

I give the model the forest photo again every frame, so it has directions home the whole time. Three seeds tell me something about starting noise on this road. I'd need other scenes to say how often this works elsewhere. And these are measurements of images. There's no measurement here of what seeing one of them would feel like.

[Full report, controls and reproduction commands](../experiments/results.md)
