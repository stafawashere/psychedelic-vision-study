# I gave the block a lattice. Where did it go?

The regular run leaves two explanations open. Maybe the first block never made a symmetric pattern. Or maybe something symmetric could be sitting there and still never make it to the pixels. I can separate those by putting a known pattern into the block and following it out.

I replaced the output of block 0 with a hexagonal lattice. I matched the spread of values in each channel so this wouldn't just be a test of making the whole thing louder. Going in, the lattice scored 1.000 on mirror symmetry with a bump of 10.10. At the image, eight seeds averaged 0.120 on signed mirror symmetry and 1.036 on the bump. Even the uninjected flat-field baseline had a larger bump, 2.099.

Then I scrambled the phases. That keeps the frequency content while breaking the alignment that makes the mirror test work. On the matched seeds the intact injection gave a mean absolute mirror reading of 0.018 and the scrambled one gave 0.071. The supposed advantage of handing the model perfect symmetry went the wrong way.

I know the layout I put in didn't survive this particular route. My explanation is that the later blocks treat it as material for building an image. They can use local features without preserving where all the hexagons sat relative to each other. I'd need measurements at more blocks to say where the layout first breaks.

This injection was a diagnostic. The forest-road runs that count toward emergence never got the lattice.

[Activation experiment](../experiments/lattice-probe.md) · [Injection controls](../experiments/results.md)
