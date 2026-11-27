# Dose curve study

Two clocks in this project. First clock = where I am in the twelve-minute arc. Second clock = how far through denoising ONE frame I am. Frame 13 is at the top of the first clock. It still has its own denoising steps to go through.

## Why frame 13?

40 frames gives me 39 gaps. Each gap is 12/39 minutes. Multiply by 13 and I land exactly at 4 minutes, where the dose reaches 1. If I picked a frame count that skips over the peak, the highest frame I measured would have milder settings than the peak I claimed to test. The runner checks this before loading the model.

The rise and fall have different shapes. Before peak I use exp((t - 4)/1.8). After peak it's exp(-(t - 4)/3) in the deliverable config. So frame 00 already has a little dose and frame 39 still has a tail. I actually have to compare the renders to see how close they get.

A hold adds time at the top. That's how I ask if the picture keeps changing once the knobs stop moving.

[Curve](../../src/core/dose.py) · [Run settings](../../configs/deliverable_fall3.yaml)
