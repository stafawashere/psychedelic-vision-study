# How do I measure spacing in a picture?

Think of bright marks about 32 pixels apart across a 512 pixel image. That's about 16 repeats across the width. In frequency space those repeats can leave power away from the center. A forest has power at loads of frequencies already, especially the low ones. So just finding the biggest raw value doesn't tell me much.

M-1 fits the background trend and looks for a peak that sticks up above it. Bump = how much it sticks up. Wavelength = where the code found that peak, converted back to pixels.

## Try it on something I know

A clean 24 px lattice reads 24.4 px. Give the instrument a broad band over a sloping background and the fitted trend can shift the leftover peak. It underestimated spacing by 20 to 50 percent in the band tests. Even the no-signal control gives a wavelength of 17.8 px. There's always somewhere for an argmax to land.

On seed 0 the peak bump was 0.99, up from 0.42 at frame 00. That's evidence of extra repetition in this run. The number 20.5 px beside it gets a lot harder to interpret when seed 1 lands at 46.5.

[Calibrations and measurement traps](experiments/measurements.md)
