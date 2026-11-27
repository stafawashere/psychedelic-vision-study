# Which run am I talking about?

"Seed 0, forest road" sounds specific until I have two seed 0 forest runs with different inputs. The older scene.jpg starts at 641 by 360 and gets resized square. The main scene.png is already 512 by 512. The captioner gets the scene path before the runner resizes the input for rendering. So checking that the resized RGB pixels match doesn't tell me that the words match too.

This actually showed up in the hold scores. The old mean change of 6.15 belonged to the JPG. Scoring the archived JPG frames again gave 6.152. The current PNG baseline across three seeds was 3.686. Looking at the old frames again was enough to separate a changed input from a broken metric.

## How I'd check a number in the paper

1. Open the resolved config beside the frames. Check the seed, caption, arm, input filename and dose settings.
2. Pick frame 00 from THAT folder as the reference. Frame 13 = peak. Frame 39 = return for these three main arcs.
3. Convert RGB to floating point before subtracting. For pixel error, scale to 0–1, square the differences, average over every pixel and channel, then take the square root.
4. Keep all three seed rows. In order the ending errors round to 0.0241, 0.0329 and 0.0262. For detail use M-2 separately.

A passing verification script tells me the implementation checks passed. The frame comparison tells me what image the run actually produced. I want both attached to the claim.

[Archived seed 0 config](../../results/seed-0/config.yaml) · [Full scoreboard](experiments/results.md)
