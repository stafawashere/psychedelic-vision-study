# What actually changes when I switch seeds?

A seed tells the generator where to start drawing random numbers. The engine starts a CPU generator from the same seed on each ordinary frame. It samples the VAE latents and draws initial noise. The scene, previous frame and controls can still change, so reusing that stream doesn't make 40 identical pictures.

Change ONLY the seed between arcs and I can see how much depends on that choice. Peak bump = 0.99, 2.69, 1.71. Peak spacing = 20.5, 46.5, 28.4 px. If I put seeds 0 and 2 side by side I might come away thinking I found a narrow range of spacings. Seed 1 belongs in the table for exactly that reason.

## Three different controls I'd want beside a score

- Known lattice = can the instrument see the thing when I KNOW it's there?
- No-signal input = how big a reading does the instrument give me for free?
- Different photo = could a completely different picture pass my test for getting the starting scene back?

The 40th frame has already inherited something from the 39th. Counting 40 frames per seed as 40 independent successes would make one run sound like a much bigger experiment than it was.

[Original frames and comparison scores](experiments/results.md)
