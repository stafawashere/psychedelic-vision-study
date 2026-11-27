# What three seeds buy me

Frame 13 looks different when I change the seed. To see how much that changes my conclusion, I keep the photo, prompt and dose curve fixed and swap the starting noise. Each row below is one whole 40-frame arm A run.

| Seed | Peak bump | Peak spacing (px) | Final pixel error | Final detail |
|---|---:|---:|---:|---:|
| 0 | 0.99 | 20.5 | 0.0241 | 0.852 |
| 1 | 2.69 | 46.5 | 0.0329 | 0.813 |
| 2 | 1.71 | 28.4 | 0.0262 | 0.872 |

Seed 1 does two interesting things at once. It has the strongest peak and the widest spacing. If I only showed seed 0 and seed 2, their spacings would both sit in the 18 to 32 pixel band. Put seed 1 back and the tidy scale story is gone. All three peak bumps still fall short of 5.

Now read the same rows from the other end. Every seed gets its pixel error below 0.05 and its detail above 0.8. The tight one is seed 1 at 0.813. Changing the noise again could reasonably change that verdict. The result I have is three passes out of three on this photo.

I also can't turn three seeds into 120 independent tests by counting every frame. Frame 14 gets some of what the model just made at frame 13. I use frames to follow the arc and seeds to describe how much the arc varies.

[Seed sweep and comparison photo](../experiments/results.md)
