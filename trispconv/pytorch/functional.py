from __future__ import annotations

from typing import Optional

import torch

from trispconv.pytorch.indices import IndiceData
from trispconv.pytorch.triton_kernels import (
    TRITON_AVAILABLE,
    gather_gemm_scatter,
)


def sparse_conv_forward(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    indice_data: IndiceData,
    N_out: int,
) -> torch.Tensor:
    """Sparse convolution forward pass.

    weight must be in internal (Kvol, C_in, C_out) layout.
    Dispatches to Triton gather-GEMM-scatter on CUDA; falls back to
    a pure-PyTorch loop on CPU or when Triton is unavailable.
    """
    orig_dtype = features.dtype
    C_out = weight.shape[2]

    use_triton = TRITON_AVAILABLE and features.device.type == "cuda"

    # Upcast to fp32 so that neither the Triton MMA accumulator nor the output
    # buffer can overflow when the original dtype is fp16/bf16.
    if use_triton and orig_dtype != torch.float32:
        features = features.to(torch.float32)
        weight = weight.to(torch.float32)

    out = torch.zeros(N_out, C_out, dtype=features.dtype, device=features.device)

    for k, (in_ids, out_ids) in enumerate(zip(indice_data.pairs_in, indice_data.pairs_out)):
        if in_ids.numel() == 0:
            continue
        if use_triton:
            gather_gemm_scatter(features, weight[k], out, in_ids, out_ids)
        else:
            out.index_add_(0, out_ids, features[in_ids] @ weight[k])

    if bias is not None:
        out = out + bias

    return out.to(orig_dtype) if out.dtype != orig_dtype else out
