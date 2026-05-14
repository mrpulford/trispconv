from __future__ import annotations

from typing import Tuple

import torch


def normalize_spconv_weight(
    w: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    in_channels: int,
    out_channels: int,
) -> torch.Tensor:
    """Convert a spconv checkpoint weight to internal (Kvol, C_in, C_out) layout.

    spconv 2.x default layout (ALL_WEIGHT_IS_KRSC=True, the shipped default):
        KRSC  ->  (C_out, kD, kH, kW, C_in)

    For kernel_size=(1,1,1) the KRSC weight is (C_out, 1, 1, 1, C_in) which
    could also look like (C_out, C_in) after squeeze — we handle both.
    """
    kD, kH, kW = kernel_size
    Kvol = kD * kH * kW

    expected_krsc = (out_channels, kD, kH, kW, in_channels)
    expected_krsc_squeezed = (out_channels, in_channels)  # Kvol==1 with squeezed dims

    if w.shape == expected_krsc:
        if Kvol == 1:
            # spconv special-cases 1x1x1: computes features @ weight.view(C_in, C_out)
            # rather than features @ W.T (the convention used for k>1).
            # weight.view(C_in, C_out) reinterprets the flat KRSC storage without permuting,
            # so we must materialise it the same way to match checkpoint-trained weights.
            return w.reshape(in_channels * out_channels).view(in_channels, out_channels).contiguous().unsqueeze(0)
        # KRSC: (C_out, kD, kH, kW, C_in) -> (Kvol, C_in, C_out)
        return w.reshape(out_channels, Kvol, in_channels).permute(1, 2, 0).contiguous()

    if Kvol == 1 and w.shape == expected_krsc_squeezed:
        # Squeezed KRSC for 1x1x1 kernels: (C_out, C_in) treated as (C_in, C_out) by spconv
        return w.reshape(in_channels * out_channels).view(in_channels, out_channels).contiguous().unsqueeze(0)

    # Also accept already-internal layout
    if w.shape == (Kvol, in_channels, out_channels):
        return w.contiguous()

    raise ValueError(
        f"Cannot identify weight layout. shape={w.shape}, "
        f"kernel_size={kernel_size}, in_channels={in_channels}, out_channels={out_channels}. "
        f"Expected KRSC {expected_krsc} or internal {(Kvol, in_channels, out_channels)}."
    )
