# How much brain is actually in the model?

Take the words "dirt road." They tell Stable Diffusion roughly what should be in the picture. They don't tell it where the little branches cross the sky. The photo carries that information. Having those two inputs means I can ask what happens when I change how much each one counts.

REBUS proposes that psychedelics loosen the influence of some high-level expectations. I borrowed one piece of that idea. I can turn down the strength of the caption while leaving the words themselves alone. The number I change is classifier-free guidance. A guidance scale of 6 means something to the denoiser. I have no conversion from 6 to a measurement of somebody's beliefs. [Carhart-Harris and Friston, 2019](https://doi.org/10.1124/pr.118.017160)

The harder prediction comes from models of visual cortex. In the model Bressloff and colleagues studied, which parts of the field interact helps determine which patterns grow. That makes a lattice a much more specific target than an image getting noisy. Where a patch sits relative to its neighbors matters. [Bressloff et al., 2001](https://doi.org/10.1098/rstb.2000.0769)

My gain hook has a simpler job. Multiply the output of one block. It can make a feature that already exists louder. There's nowhere in that multiplication to say "put the next bright patch 24 pixels to the left." If a particular spacing shows up, it has to come from somewhere else in the system. That's why I wanted one arm with no map added and a second arm where I tell you exactly what spatial assumption I put in.

[Model settings](../../configs/deliverable_fall3.yaml) · [Arm definitions and measurements](../experiments/results.md)
