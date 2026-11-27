# Four ways to look at the same frame

If the road goes blurry, I can still tell you it's a road. If I use recognition as my test I'll probably give that frame a pass. I needed a score that cares where the edges went.

| Measure | Question I can ask with it | A check on the answer |
|---|---|---|
| M-1, structure | Does one spacing stick out above the background spectrum? Does the pattern line up with a flipped or rotated copy? | The known hex lattice scores 14.54 on the bump. The forest photo scores 0.44. |
| M-2, detail | Do the small edges point in the same directions at the same places as in the reference? | Compare frame 00 to itself and I get 1.000. Blur away the detail and the score drops. |
| M-3, motion | With the controls held still, how much do adjacent frames change? Can I still line up their patterns? | Random churn can change a lot too. Its peak correlation in the hold control was only 0.158. |
| M-4, return | How far apart are the RGB pixels, and does the fine detail line up too? | A beach scored about zero on detail against the starting forest render. |

M-1 has a trap. It can report a wavelength even on the no-signal control. So "the strongest spacing is 28 pixels" means very little on its own. I look at how strong the peak is, then ask what happens if I flip or rotate the image. Even a strong peak can't answer that second question.

M-2 has a different trap. Which image am I comparing to? A photo has fine structure that the first pass through the model already changes. For the return I use the first render. For the early-detail target the plan left the reference open, so the paper gives both readings. Picking whichever one looks better after the run would change the experiment.

[Instrument definitions and controls](../experiments/measurements.md) · [Recorded scores](../experiments/results.md)
