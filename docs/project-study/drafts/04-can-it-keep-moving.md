# What happens if I hold the dose at the top?

## Question

A still frame can make a pretty convincing peak. I wanted to know what happens one frame later. If the dose stops changing, does the model keep building on what it sees? And when I let the dose fall again, does it get back to the road it started from? For breakthrough and return to count together I need both in the same run.

## Five knobs

My input is a forest photo and Stable Diffusion 1.5's job is to render it. `scene_precision` sets how much it listens to the photo. `prior_precision` sets how much it listens to the caption. `recurrence` feeds part of the previous render back in, so something the model made one frame ago can become input this time. `depth` picks the UNet block I tap and `gain` turns up what that block is already doing. Depth stays shallow here. One twelve-minute dose curve controls the arc.

## Rules

Arm A gets a neutral caption, no warp and no geometry drawn in. Some of the photo always gets through. I wrote down the limits first. For a separate motion test I let the starting noise drift in a repeatable way. That adds something outside the five knobs, so any motion I get from it belongs to that test.

## Scoreboard

In the ordinary arc the fine detail drops out too soon. Repetition peaks at a mean of 1.80 over three seeds against a target above 5, and the mirror and rotation scores stay near zero. That's three misses already. Code integrity passed. Motion and return are where the two setups split.

| Arm A setup | What happens during a hold at peak? | Where does a full arc end? |
|---|---|---|
| Ordinary starting noise | Change dies down at 3 of 3 seeds. Mean change 3.686 | 3 of 3 close. Mean RMSE 0.0277, detail match 0.846 |
| Drifting starting noise | Mean change rises to 31.726. Only 2 of 3 seeds sustain it | Tested arc misses both limits. RMSE 0.1498, detail match 0.028 |

## Finding

Here's the part I would've missed by only watching the video. With drift on, the last frame still looks like a forest road. But I started the noise at one place and kept walking it forward. By frame 39 I'm feeding the model different noise than I fed it at frame 00. It draws me another version of that road. Put the fine details side by side and they barely line up.

## Limits

The ordinary setup passed return and missed sustained motion. Drift made the peak move more, but one of three holds still decayed, and the tested drift arc missed return. I can't take the moving peak from one video and the ending from the other and say I got the full thing. This tradeoff is specific to how I moved the noise. It tells me what happened in these renders, nothing about what a person experiences at a peak.

[Full report and drift controls](../experiments/results.md)
