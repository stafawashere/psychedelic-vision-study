# I put a lattice into the first UNet block. What came out?

August 29, 2026. Eight renders with the lattice and three with its phases scrambled.

At the peak I kept getting a texture with some spacing to it. Looking at the final image, I couldn't tell where the geometry got lost. Maybe the first block never made anything organized. Or maybe it could have a perfect pattern sitting in it and the rest of the network would still draw something else. I can test that second idea directly. Give the block a pattern whose spacing and symmetry I know, then measure those same things in the pixels.

## What I put in

I used a flat gray image and an empty caption. At block 0 I replaced the forward output, `output[0]`, with a hexagonal lattice at every denoising step. Each channel kept its own mean and standard deviation. The skip copies went straight across as usual. This was a temporary PyTorch hook. The ordinary forest runs never got this lattice.

The lattice repeats every four cells on a 32 by 32 feature map. Scale that up 16 times and its period is 64 pixels. Measured on the field going IN, the mirror score was 1.000 and the spectral bump was 10.10. I had a clean pattern to start from.

| Setting | Value |
|---|---|
| Denoise strength and steps | 0.82, 30 steps |
| Scene precision, prior precision, recurrence | 0.85, 1.0, 0.5 |
| Gain and clipping | 1.0, soft clip on |
| Measured activation SD at block 0 | 0.6495 |

## What reached the image?

| seed | wavelength | bump | mirror_lr | image std |
|---|---|---|---|---|
| 0 | 16.0 px | 1.17 | -0.051 | 0.1194 |
| 1 | 21.3 px | 0.50 | 0.537 | 0.0805 |
| 2 | 18.3 px | 1.03 | 0.396 | 0.0694 |
| 3 | 32.0 px | 0.95 | 0.037 | 0.0819 |
| 4 | 36.6 px | 1.72 | 0.006 | 0.0263 |
| 5 | 12.2 px | 0.67 | -0.012 | 0.0733 |
| 6 | 64.0 px | 1.13 | -0.112 | 0.0356 |
| 7 | 64.0 px | 1.12 | 0.163 | 0.0487 |

Look at seed 1. A mirror score of 0.537 sounds promising until I look at the frame. I see a washed-out cloud. I first thought low contrast was driving the big mirror readings. Seed 4 is a problem for that explanation. It has the lowest image SD of all eight and a mirror score of 0.006. Across seeds the explanation didn't hold up.

The more useful comparison is seeds 3, 4 and 5. I scrambled the lattice phases so the input kept its frequency content and lost its mirror alignment. On those same seeds the intact lattice averaged about 0.018 in absolute mirror score. The scrambled input gave about 0.071. Preserving the mirror axis on the way IN gave me no advantage on the way OUT.

Re-scoring the eight original PNGs with M-1 gives mean signed mirror 0.120 (sample SD 0.231) and mean bump 1.036 (sample SD 0.363). I used the full pixel data before rounding. For seeds 3, 4 and 5 the mean absolute mirror is 0.018 for the intact lattice and 0.071 for the scrambled one. An earlier summary used different aggregates. The per-seed rows above are rounded for display. The output spacing ranges from 12.2 to 64.0 px. Two seeds hit 64, which is also the instrument's ceiling.

## What I think happened

I know where the lattice went in. I know the final images failed to preserve its layout. My explanation is that the later UNet blocks and VAE use the activations to build the image again. They can respond to how strong a feature is without keeping every patch in the place I put it. I'd need measurements between blocks to locate the loss.

That changed what seemed worth building. C-1, C-2, C-5, C-11 and E-7 all depended on imposing something inside the activations. I retired those routes. E-8 changes the starting noise. C-3 changes the guidance signal. C-4 changes gain over time. Those ideas touch different parts of the process and needed their own decisions. The [development plan](design.md) records what happened next.

The test covers one block, one lattice family and one injection schedule. Turning up gain at that same block clearly changes the image. What failed here was getting a known spatial layout through to the pixels.

[Archived injection and phase-scramble PNGs](../../../results/lattice-probe/)
