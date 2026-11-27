# How close does it have to get for me to call it back?

## Question

If I show you two pictures of a dirt road through a forest, you'd probably describe them with the same sentence. The trees could be in totally different places. So when I say the scene "comes back" after a simulated dose, I need something better than the caption to check that. I also need to show it went somewhere in the first place.

## Five knobs

I gave a frozen Stable Diffusion 1.5 the road photo and a caption describing it. `scene_precision` is the weight on what it sees in the photo. `prior_precision` is the weight on what it expects from the words. Both drop as the dose rises. `recurrence` turns up how much of the last frame gets another pass. The other pair work together inside the UNet. `depth` picks the block, `gain` multiplies what comes out of it. For this test I leave depth at the first block and let gain rise and fall. Everything follows one dose curve with a peak at four minutes.

## Rules

Arm A has no geometry prompt, no hand-drawn pattern and no spatial warp. The photo weight stays above zero even at peak. I set two limits for return before the run. Compare the last frame to the first render. Pixel error has to come in at or below 0.05 AND fine-detail match has to reach 0.8. A miss on either one counts. Keep the seed either way.

## Scoreboard

| Arm A return, steady photo | Pixel error | Fine-detail match |
|---|---:|---:|
| Seed 0 | 0.0241 | 0.852 |
| Seed 1 | 0.0329 | 0.813 |
| Seed 2 | 0.0262 | 0.872 |
| Different photo, same comparison across seeds | 0.4427 to 0.4531 | -0.001 |

All three make it back. Seed 1 only clears the detail cutoff by 0.013. For the departure check, seed 0 hit 0.2766 pixel error at the peak. By then there was very little of the starting detail left to line up.

The rest of my scorecard is rougher. Fine detail needed to last through frame 11 and was already below target at frame 06 on seed 0. The peak's repetition score averaged 1.80 across seeds, below the target of 5, and the pattern had little mirror or rotational symmetry. Motion died down when the peak was held. Return and the integrity checks were the two passes.

## Finding

There's a stretch in the middle where the image takes on a spacing of its own. Even on a flat gray input the model makes repeating texture, so it doesn't need a row of trees to copy that spacing from. But the knobs don't tell it how far apart two features should be or what point they should wrap around. My read is that I can drive the model into a texture its own layers support. I didn't get the organized rings or lattices I was looking for.

## Limits

Three starting seeds, one road. Also, the model gets to look at that same photo again on every frame. That makes returning easier and leaves a real question about a moving scene. I can say how closely these three renders came back, and how far the peak fell short of the geometry test. An image score can't tell me whether any of this resembles the experience of a trip.

[Full report and seed-by-seed results](../experiments/results.md)
