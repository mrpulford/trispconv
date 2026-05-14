from __future__ import annotations

from typing import Optional

import torch

from trispconv.pytorch.indices import IndiceData


def sparse_conv_forward(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    indice_data: IndiceData,
    N_out: int,
) -> torch.Tensor:
    """PyTorch-only sparse convolution forward pass.

    weight must be in internal (Kvol, C_in, C_out) layout.
    """
    C_out = weight.shape[2]
    out = torch.zeros(N_out, C_out, dtype=features.dtype, device=features.device)

    for k, (in_ids, out_ids) in enumerate(zip(indice_data.pairs_in, indice_data.pairs_out)):
        if in_ids.numel() == 0:
            continue
        gathered = features[in_ids]       # (N_k, C_in)
        partial = gathered @ weight[k]    # (N_k, C_out)
        out.index_add_(0, out_ids, partial)

    if bias is not None:
        out = out + bias
    return out
