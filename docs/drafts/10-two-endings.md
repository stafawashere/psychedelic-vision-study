# Two ways to end up looking at a road

I can keep the knobs still at peak and watch the loop settle. With the ordinary starting noise, the measured frame-to-frame change dies down at all three seeds. So I tried moving the noise itself. I declared that separately because otherwise I'd be giving five knobs credit for a sixth thing I did.

On the twelve-frame holds the mean change went from 3.686 with drift off to 31.726 with drift on, three seeds per setting. Those numbers are grayscale change on a 0 to 255 scale. The peak correlation stayed at 0.8922 with drift, well above the churn control at 0.158. Adjacent frames still shared a pattern. The change held up over time at two seeds and fell off at the third. That's exactly why I need both the amount of change and its slope.

Now let the dose come down. The drift run ends looking like a forest road. Its pixel error against frame 00 is 0.1498 and its fine-detail match is 0.028. Both miss. The starting frame used the first noise anchor. By the last frame the noise walk was at position 8.153. Same photo coming in, different noise to denoise from. Another sample of the road is a perfectly reasonable thing for this setup to draw.

There's a tempting edit I could make to the story. Take the moving peak from the drift run and the closing frame from the ordinary run. The video I just described doesn't belong to either config. My original test asked one config to do both, so the paper has to keep those endings attached to the runs that produced them.

[Hold experiment and drift arc](../experiments/results.md) · [Acceptance criteria](../experiments/design.md)
