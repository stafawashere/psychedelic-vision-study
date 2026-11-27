"""Probe: does a scalar 'gain' on a block output actually change the U-Net prediction?

Why: the gain knob is the heart of the experiment (DeepDream-style activation scaling).
Online activation-steering advice says "multiply, then renormalize back to the original
norm to avoid blow-up." This probe tests whether that — and other formulations — actually
move the U-Net's final output, by hooking a block, applying each formulation, and
measuring the relative L2 change in the final unet(...).sample.

Run: .venv/bin/python testing/probe_gain_formulations.py

Findings (diffusers 0.38.0, gain=1.5), rel.Δ of final U-Net output:
    block idx 0 (shallow down):  flat_multiply 0.1314 | flat+global_renorm 0.0000 |
                                 contrast_about_mean 0.1315 | per_channel_contrast 0.1290
    block idx 2 (deep down):     flat 0.0154 | renorm 0.0000 | ...
    block idx 4 (mid bottleneck):flat 0.0095 | renorm 0.0000 | ...
    block idx 8 (final up):      ALL 0.0000

Two hard conclusions for hooks.py:
  1. "multiply then renorm-to-original-norm" == IDENTITY (0.0000 everywhere). Algebra:
     (x*g) * ||x|| / ||x*g|| == x. So NEVER stabilize gain by renorm-to-original — it
     nulls the knob. Stabilize by CLAMP / soft-clip instead.
  2. GroupNorm (start of every ResNet) erases a uniform scalar, so the gain effect decays
     with depth (0.131 -> 0.015 -> 0.0095) and is fully normalized away at the final
     pre-output block (0.0000). Hook shallow blocks for strength; don't hook the last one.
  contrast_about_mean ~= flat_multiply (they differ only by a DC offset GroupNorm also
  eats), so the plain `out*gain` the docs specify is faithful — no mean-centering needed.

Caveat: this small UNet has 32 channels == norm_num_groups, so per-channel GroupNorm is
near-InstanceNorm and per_channel_contrast looks ~= global. Real SD1.5 has 10 ch/group, so
per-channel amplification would survive somewhat more than shown here.
"""
from __future__ import annotations
import torch
from diffusers import UNet2DConditionModel

GAIN = 1.5
PROBE_INDICES = [0, 2, 4, 8]  # shallow down, deep down, mid bottleneck, final up
FORMULATIONS = [
    "flat_multiply (out*g)",
    "flat+global_renorm",
    "contrast_about_mean",
    "per_channel_contrast",
]


def build_small_unet() -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(32, 64, 64, 64),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=32,
        norm_num_groups=32,
    ).eval()


def formulations(x: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = x.mean()
    per_ch_mean = x.mean(dim=(2, 3), keepdim=True)
    amplified = x * GAIN
    return {
        "flat_multiply (out*g)": amplified,
        # rescale the multiplied tensor back to the original norm — the popular advice
        "flat+global_renorm": amplified * (x.norm() / (amplified.norm() + 1e-8)),
        # amplify deviations from the global mean
        "contrast_about_mean": mean + GAIN * (x - mean),
        # amplify deviations from each channel's own mean
        "per_channel_contrast": per_ch_mean + GAIN * (x - per_ch_mean),
    }


def scale_main(out, fn):
    # down-blocks return (hidden_states, res_samples); scale the forward path only
    if isinstance(out, tuple):
        return (fn(out[0]), out[1])
    return fn(out)


def main() -> None:
    torch.manual_seed(0)
    unet = build_small_unet()
    blocks = [*unet.down_blocks, unet.mid_block, *unet.up_blocks]

    sample = torch.randn(1, 4, 16, 16)
    timestep = torch.tensor([10])
    encoder_hidden_states = torch.randn(1, 77, 32)

    with torch.no_grad():
        base = unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample

    for idx in PROBE_INDICES:
        print(f"\nblock idx {idx} ({type(blocks[idx]).__name__}):")
        for name in FORMULATIONS:
            def hook(module, inp, out, _name=name):
                return scale_main(out, lambda x: formulations(x)[_name])

            handle = blocks[idx].register_forward_hook(hook)
            with torch.no_grad():
                out = unet(
                    sample, timestep, encoder_hidden_states=encoder_hidden_states
                ).sample
            handle.remove()

            rel = (out - base).norm().item() / base.norm().item()
            print(f"    {name:<28} rel.Δ final output = {rel:.4f}")


if __name__ == "__main__":
    main()
