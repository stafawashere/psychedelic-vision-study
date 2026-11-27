# Walking between noise samples

I can make the starting noise different every frame by picking a new seed. That makes the frames jump. What I wanted to test was a path through noise, so the next sample still has something to do with the last one.

## Why slerp?

Take two independent noise tensors A and B, each with variance about 1. Halfway on a straight line I get 0.5A + 0.5B. Its variance is about 0.25 + 0.25 = 0.5. The standard deviation falls to about 0.71. So I'd accidentally change TWO things going from one anchor to the other. The noise pattern and how strong it is.

Slerp follows the angle between the anchors. For these high-dimensional draws the noise scale stays close to the starting scale while the pattern turns toward the next one.

## A closed path doesn't guarantee a closed arc

The noise walk cycles through three anchors. Frame index times phase gives its position. At frame 39 that's 39 times 0.209062, about 8.153. Wrap it around three anchors and I'm between the last anchor and the first. I started ON the first. Letting the dose fall didn't rewind the noise walk.

Now the drift ending makes sense. Still a forest, but fine-detail match only 0.028. I have to name the noise walk whenever I use that result. It's an extra intervention.

[Noise walk code](../../../src/render/engine.py) · [Drift comparison](../../experiments/results.md)
