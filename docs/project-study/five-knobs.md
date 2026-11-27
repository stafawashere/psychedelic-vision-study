# One number goes in. What comes out?

At peak D = 1. I can plug that into the mapping before I even load a photo.

| Knob | Peak | Where that number goes |
|---|---:|---|
| scene precision | 0.40 | weight on the photo latent |
| prior precision | 6 | multiplier on caption guidance |
| gain | 2.8 | multiply the selected block's activations before soft clipping |
| depth | 0 | index of the block I hooked |
| recurrence | 0.12 | weight on the previous render's latent |

## A detail I could easily read wrong

0.40 doesn't mean 40 percent of the final pixels are forest. Even at the blend step I have to divide by the sum of the weights. At peak that gives the photo 0.40/0.52, about 77 percent of the starting blend. Then I add noise and run the whole denoiser over it. Plenty of photo can go IN and very little recognizable road can come OUT.

Depth has the same starting and ending index in this config. Four knobs actually move. Denoise strength also rides the curve as a rendering setting, so turning the curve up changes several things together. If I want to know what gain did by itself I need a separate comparison.

[Knob mapping](../../src/core/knobs.py) · [Actual blend](../../src/render/engine.py)
