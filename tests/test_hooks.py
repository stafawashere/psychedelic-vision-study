"""Validate hooks.py end-to-end on a small SD1.5-topology U-Net (no model download).

Checks, per hooked depth:
  * attach() runs without raising and produces NO NaN/Inf in the final U-Net output,
  * the gain actually MOVES the output (rel.Δ > 0) where the maths says it should,
  * detach() fully restores the baseline output (handles really removed),
  * tuple-returning down_blocks AND bare-tensor mid/up_blocks are both handled,
  * the arm-B logpolar=True warp path runs and stays finite,
  * gain == 1.0 is a near no-op (sanity; engine skips attach at gain==1 anyway).

Run: .venv/bin/python tests/test_hooks.py
"""
from __future__ import annotations
import os
import sys
import torch
from diffusers import UNet2DConditionModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.render import hooks


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


def main() -> None:
    torch.manual_seed(seed=0)
    unet = build_small_unet()
    blocks = hooks.hookable_blocks(unet)
    assert len(blocks) == 9, f"expected 9 hookable blocks, got {len(blocks)}"

    sample = torch.randn(1, 4, 16, 16)
    timestep = torch.tensor([10])
    ehs = torch.randn(1, 77, 32)

    def run() -> torch.Tensor:
        with torch.no_grad():
            return unet(sample, timestep, encoder_hidden_states=ehs).sample

    base = run()
    failures = []

    def rel(out: torch.Tensor) -> float:
        return (out - base).norm().item() / base.norm().item()

    print(f"{'depth':<6}{'block':<26}{'gain':<6}{'warp':<6}{'finite':<8}{'rel.Δ':<10}{'restored'}")
    
    for depth in range(9):  # every index, tuple-return AND bare-tensor blocks
        for gain, warp in [(2.0, False), (1.0, False)]:
            handles = hooks.attach(unet, depth, gain, logpolar=warp)
            out = run()
            hooks.detach(handles)
            restored = run()

            finite = bool(torch.isfinite(input=out).all())
            is_restored = torch.allclose(input=restored, other=base, atol=1e-6)
            r = rel(out)
            print(f"{depth:<6}{type(blocks[depth]).__name__:<26}{gain:<6}{str(warp):<6}"
                  f"{str(finite):<8}{r:<10.4f}{is_restored}")

            if not finite:
                failures.append(f"depth {depth} gain {gain}: non-finite output")
            if not is_restored:
                failures.append(f"depth {depth} gain {gain}: detach did not restore baseline")
            if gain == 1.0 and r > 0.05:
                failures.append(f"depth {depth}: gain=1.0 moved output by {r:.3f} (should be ~0)")

    # arm-B log-polar warp path: must run and stay finite on tuple + bare-tensor blocks
    print("\nlog-polar (arm B) warp path:")
    for depth in [0, 4, 8]:
        handles = hooks.attach(unet, depth=depth, gain=2.0, logpolar=True)
        out = run()
        hooks.detach(handles)
        finite = bool(torch.isfinite(out).all())
        print(f"  depth {depth:<2} ({type(blocks[depth]).__name__:<22}) finite={finite}  rel.Δ={rel(out):.4f}")
        if not finite:
            failures.append(f"logpolar depth {depth}: non-finite output")

    # gain-mode selector (E-4): residual mode must be a REAL alternative on down blocks and a
    # true no-op on bare-tensor blocks, never a silent fallback to forward mode
    print("\ngain-mode selector:")
    for depth in [0, 3, 4, 8]:
        deltas = {}
        for mode in hooks.GAIN_MODES:
            handles = hooks.attach(unet, depth=depth, gain=2.8, logpolar=False, soft=True, mode=mode)
            out = run()
            hooks.detach(handles)
            deltas[mode] = rel(out)
            if not bool(torch.isfinite(out).all()):
                failures.append(f"mode {mode} depth {depth}: non-finite output")
            if not torch.allclose(input=run(), other=base, atol=1e-6):
                failures.append(f"mode {mode} depth {depth}: detach did not restore baseline")

        is_down_block = isinstance(blocks[depth], tuple(type(b) for b in unet.down_blocks))
        print(f"  depth {depth:<2} ({type(blocks[depth]).__name__:<22}) "
              + "  ".join(f"{m}={deltas[m]:.4f}" for m in hooks.GAIN_MODES))

        if is_down_block:
            if deltas["residual"] <= 0.0:
                failures.append(f"residual mode is a no-op on down block {depth}")
            if abs(deltas["residual"] - deltas["forward"]) < 1e-6:
                failures.append(f"residual mode matches forward mode on down block {depth}")
            # residual must also stay distinct from both, or it is amplifying the forward path too
            if abs(deltas["residual"] - deltas["both"]) < 1e-6:
                failures.append(f"residual mode matches both mode on down block {depth}")
        else:
            if deltas["residual"] != 0.0:
                failures.append(
                    f"residual mode moved bare-tensor block {depth} by {deltas['residual']:.4f}; "
                    "those blocks have no residual stream and must be untouched"
                )
            if abs(deltas["both"] - deltas["forward"]) > 1e-6:
                failures.append(f"both mode differs from forward on bare-tensor block {depth}")

    # an unknown mode must raise rather than falling through to forward
    try:
        hooks.attach(unet, depth=0, gain=2.0, logpolar=False, mode="apical")
        failures.append("unknown gain mode did not raise ValueError")
    except ValueError:
        pass

    # out-of-range depth must raise, not silently no-op
    try:
        hooks.attach(unet, depth=9, gain=2.0, logpolar=False)
        failures.append("depth 9 did not raise IndexError")
    except IndexError:
        pass

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("all hooks.py checks passed")


if __name__ == "__main__":
    main()
