# Can I guide the small detail separately?

Normal guidance gives me a whole tensor for "what changed when I added the caption." One scale multiplies the whole thing. The branches are small and the road layout is big. Maybe I could keep a grip on one without dragging the other around with it.

Blur the guidance direction and call that the low part. Subtract it from the original direction and I've got what's left over, the high part. Now put them back together with a weight on the high part.

> low + high = original direction. high ratio 1 is the check that I can put the pieces back.

## Does changing the tensor change the image?

I tried high ratios of 0, 1 and 2, three seeds at each setting. The image's high-band energy fraction went 0.740, 0.794, 0.815. That's in the direction I expected. But I needed the endpoint gap to be three times the pooled within-setting SD. I got 1.58 times.

There's another catch. More high-frequency energy could just mean I made more tiny marks. M-2 asks whether those marks line up with the actual branches. Its scores stayed between 0.008 and 0.013 across all nine frames. The goal was 0.6. That small probe already answered the question I needed for the longer 66-frame run.

[Guidance split](../../../src/render/engine.py) · [C-3 decision](../../experiments/design.md)
