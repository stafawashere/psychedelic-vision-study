# Why turning a block up can do almost nothing

Imagine a row of activations. Multiply every number by two. The spread doubles too. If the next operation subtracts the mean and divides by the spread, most of that factor of two cancels. That's the problem with expecting a plain multiply to keep getting louder all the way through a normalized network.

My hook has another piece. It measures each channel's spatial standard deviation BEFORE multiplying, puts a bound at six times that spread, and soft-clips with tanh. Small values near the middle can still move a lot. Big values start crowding near the bound. Now I've changed the relative sizes inside the channel. Normalizing the result can't simply undo that.

## What if I amplify the skip photocopies?

I tried the residual route. Image SD dropped to 0.0028 against 0.1574 on the forward route in that probe. Almost the whole frame went the same color. Amplifying both routes gave mean bumps of 2.13 and 2.12 for both versus forward alone. The extra route bought me very little.

Depth 8 is back near full resolution. It's close to the final noise prediction too. A stronger intervention there can wipe out the picture before I ever get to ask whether the geometry improved.

[Gain hook](../../src/render/hooks.py) · [Gain experiments](experiments/design.md)
