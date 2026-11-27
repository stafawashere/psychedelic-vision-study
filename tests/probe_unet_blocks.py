"""Probe: what does a forward hook actually receive from each SD1.5 U-Net block?

Why: hooks.py must scale a block's activations via register_forward_hook, but in
diffusers the block return types are NOT uniform — down-blocks return a
(hidden_states, res_samples) tuple while mid/up-blocks return a bare tensor. Returning
the wrong shape from the hook breaks UNet2DConditionModel.forward's unpack.

This builds a *small* UNet with the SAME block topology as SD1.5 (the types decide the
return signature; channel sizes don't), registers a describe-hook on every block in the
flat [*down_blocks, mid_block, *up_blocks] list, and prints what each hook sees.

Run: .venv/bin/python testing/probe_unet_blocks.py

Findings (diffusers 0.38.0, torch 2.12.1):
    idx 0 CrossAttnDownBlock2D   -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]
    idx 1 CrossAttnDownBlock2D   -> tuple[...]
    idx 2 CrossAttnDownBlock2D   -> tuple[...]
    idx 3 DownBlock2D            -> tuple[Tensor, tuple[Tensor, Tensor]]
    idx 4 UNetMidBlock2DCrossAttn-> Tensor                      (bare)
    idx 5 UpBlock2D              -> Tensor                      (bare)
    idx 6 CrossAttnUpBlock2D     -> Tensor                      (bare)
    idx 7 CrossAttnUpBlock2D     -> Tensor                      (bare)
    idx 8 CrossAttnUpBlock2D     -> Tensor                      (bare)
=> the hook must branch on isinstance(output, tuple): scale output[0] for down-blocks,
   scale the tensor directly for mid/up-blocks.

For real SD1.5 (block_out_channels=(320,640,1280,1280), 512x512 -> 64x64 latent) the
main-output (channels, H, W) of each block in the flat list is:
    0:320x32x32  1:640x16x16  2:1280x8x8  3:1280x8x8  4:1280x8x8
    5:1280x16x16 6:1280x32x32 7:640x64x64 8:320x64x64
"""
from __future__ import annotations
import torch
from diffusers import UNet2DConditionModel


def build_small_unet() -> UNet2DConditionModel:
    # same block topology as SD1.5, shrunk so it builds instantly with no download
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


def describe(x) -> str:
    if isinstance(x, torch.Tensor):
        return f"Tensor{tuple(x.shape)}"
    if isinstance(x, (tuple, list)):
        return f"{type(x).__name__}[" + ", ".join(describe(e) for e in x) + "]"
    return type(x).__name__


def main() -> None:
    unet = build_small_unet()
    blocks = [*unet.down_blocks, unet.mid_block, *unet.up_blocks]

    print("FLAT BLOCK LIST [*down_blocks, mid_block, *up_blocks]:")
    for i, b in enumerate(blocks):
        print(f"  idx {i}: {type(b).__name__}")
    print()

    def make_hook(idx: int, name: str):
        def hook(module, inp, out):
            print(f"  idx {idx:>1} {name:<24} OUTPUT -> {describe(out)}")
        return hook

    handles = [
        b.register_forward_hook(make_hook(i, type(b).__name__))
        for i, b in enumerate(blocks)
    ]

    sample = torch.randn(1, 4, 16, 16)
    timestep = torch.tensor([10])
    encoder_hidden_states = torch.randn(1, 77, 32)  # text conditioning

    print("HOOK OUTPUT TYPES PER BLOCK (one forward pass):")
    with torch.no_grad():
        unet(sample, timestep, encoder_hidden_states=encoder_hidden_states)

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
