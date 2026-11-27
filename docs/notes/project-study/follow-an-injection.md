# Can I send a shape through a hook?

In the ordinary render I can't look at the final texture and tell whether block 0 ever had a lattice in it. So for this probe I put one there myself. Hex lattice, four cells per repeat. At block 0 that's 64 image pixels. Make the channels match their usual mean and spread, replace the forward output at each denoise step, leave the skip copies alone.

The input mirror score was 1.000. At seed 1 the image scored 0.537. Easy to read that as "some lattice got through." Except the frame looked like a low-contrast cloud. A useful control here has to keep the spacing while messing up the alignment.

## Phase scramble

In frequency space the magnitude says how much of each frequency I have. Phase helps decide where those pieces line up. Scramble phase and I can keep the frequency content while wrecking the mirror layout. On matched seeds 3, 4 and 5 the intact input gave about 0.018 absolute mirror at the image. The scrambled version gave about 0.071. Starting with the right alignment didn't help.

Re-scoring the original eight PNGs gives a mean signed mirror of 0.120. On the matched seeds the scrambled version gave about 0.071, using the unrounded scores. I can say where I put the pattern in and where I measured it coming out. The exact block where it fell apart is still an open question.

[Injection settings, seed rows and discrepancy](../../experiments/lattice-probe.md)
