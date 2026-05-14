from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List, Tuple

import torch


@dataclass
class IndiceData:
    out_indices: torch.Tensor           # (N_out, 4) int32
    pairs_in: List[torch.Tensor]        # len K, each (N_k,) long
    pairs_out: List[torch.Tensor]       # len K, each (N_k,) long
    pair_num: torch.Tensor              # (K,) number of active pairs per kernel slot
    input_indices: torch.Tensor         # (N_in, 4) int32 — original input coords
    input_spatial_shape: Tuple[int, ...]
    output_spatial_shape: Tuple[int, ...]
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    subm: bool


def _to3(x) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    return tuple(int(v) for v in x)


def _output_spatial_shape(
    in_shape: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    out = []
    for s, k, st, p, d in zip(in_shape, kernel_size, stride, padding, dilation):
        out.append((s + 2 * p - d * (k - 1) - 1) // st + 1)
    return tuple(out)


def _encode_coords(
    indices: torch.Tensor,
    D: int, H: int, W: int,
) -> torch.Tensor:
    """Map (b, z, y, x) int32 coords to unique int64 scalars."""
    idx = indices.long()
    return idx[:, 0] * (D * H * W) + idx[:, 1] * (H * W) + idx[:, 2] * W + idx[:, 3]


def _build_hash_table(
    indices: torch.Tensor,
    D: int, H: int, W: int,
    batch_size: int,
) -> torch.Tensor:
    """Return a lookup tensor: encoded_coord -> row_id, -1 if absent."""
    size = batch_size * D * H * W
    table = torch.full((size,), -1, dtype=torch.long, device=indices.device)
    encoded = _encode_coords(indices, D, H, W)
    table[encoded] = torch.arange(len(indices), dtype=torch.long, device=indices.device)
    return table


def generate_subm_indice_pairs(
    indices: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    batch_size: int,
    kernel_size,
    dilation=1,
) -> IndiceData:
    """Submanifold: output active set == input active set."""
    kD, kH, kW = _to3(kernel_size)
    dz, dy, dx = _to3(dilation)
    D, H, W = spatial_shape
    Kvol = kD * kH * kW
    N = len(indices)

    table = _build_hash_table(indices, D, H, W, batch_size)

    pairs_in: List[torch.Tensor] = []
    pairs_out: List[torch.Tensor] = []
    pair_num: List[int] = []

    iz = indices[:, 1].long()
    iy = indices[:, 2].long()
    ix = indices[:, 3].long()
    ib = indices[:, 0].long()

    for kz in range(kD):
        for ky in range(kH):
            for kx in range(kW):
                # offset relative to kernel center
                offz = (kz - kD // 2) * dz
                offy = (ky - kH // 2) * dy
                offx = (kx - kW // 2) * dx

                nz = iz + offz
                ny = iy + offy
                nx = ix + offx

                valid = (nz >= 0) & (nz < D) & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
                vidx = valid.nonzero(as_tuple=True)[0]

                if vidx.numel() == 0:
                    pairs_in.append(torch.empty(0, dtype=torch.long, device=indices.device))
                    pairs_out.append(torch.empty(0, dtype=torch.long, device=indices.device))
                    pair_num.append(0)
                    continue

                enc = ib[vidx] * (D * H * W) + nz[vidx] * (H * W) + ny[vidx] * W + nx[vidx]
                in_ids = table[enc]
                found = in_ids >= 0

                if not found.any():
                    pairs_in.append(torch.empty(0, dtype=torch.long, device=indices.device))
                    pairs_out.append(torch.empty(0, dtype=torch.long, device=indices.device))
                    pair_num.append(0)
                    continue

                pairs_in.append(in_ids[found])
                pairs_out.append(vidx[found])
                pair_num.append(found.sum().item())

    return IndiceData(
        out_indices=indices,
        pairs_in=pairs_in,
        pairs_out=pairs_out,
        pair_num=torch.tensor(pair_num, dtype=torch.long),
        input_indices=indices,
        input_spatial_shape=spatial_shape,
        output_spatial_shape=spatial_shape,
        kernel_size=_to3(kernel_size),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=_to3(dilation),
        subm=True,
    )


def generate_strided_indice_pairs(
    indices: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    batch_size: int,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
) -> IndiceData:
    """Strided sparse convolution: generate new output active set."""
    kD, kH, kW = _to3(kernel_size)
    sz, sy, sx = _to3(stride)
    pz, py, px = _to3(padding)
    dz, dy, dx = _to3(dilation)
    D, H, W = spatial_shape
    out_D, out_H, out_W = _output_spatial_shape(
        spatial_shape,
        (kD, kH, kW),
        (sz, sy, sx),
        (pz, py, px),
        (dz, dy, dx),
    )
    Kvol = kD * kH * kW
    device = indices.device

    iz = indices[:, 1].long()
    iy = indices[:, 2].long()
    ix = indices[:, 3].long()
    ib = indices[:, 0].long()

    all_in_ids: List[torch.Tensor] = []
    all_out_coords: List[torch.Tensor] = []
    all_k_ids: List[torch.Tensor] = []

    for k, (kz, ky, kx) in enumerate(product(range(kD), range(kH), range(kW))):
        cz = iz + pz - kz * dz
        cy = iy + py - ky * dy
        cx = ix + px - kx * dx

        valid = (cz % sz == 0) & (cy % sy == 0) & (cx % sx == 0)
        oz = cz // sz
        oy = cy // sy
        ox = cx // sx

        valid &= (oz >= 0) & (oz < out_D) & (oy >= 0) & (oy < out_H) & (ox >= 0) & (ox < out_W)

        vidx = valid.nonzero(as_tuple=True)[0]
        if vidx.numel() == 0:
            continue

        all_in_ids.append(vidx)
        all_out_coords.append(torch.stack([ib[vidx], oz[vidx], oy[vidx], ox[vidx]], dim=1))
        all_k_ids.append(torch.full((vidx.numel(),), k, dtype=torch.long, device=device))

    if not all_in_ids:
        empty = torch.empty(0, dtype=torch.long, device=device)
        zero_coords = torch.empty((0, 4), dtype=torch.int32, device=device)
        return IndiceData(
            out_indices=zero_coords,
            pairs_in=[empty] * Kvol,
            pairs_out=[empty] * Kvol,
            pair_num=torch.zeros(Kvol, dtype=torch.long),
            input_indices=indices,
            input_spatial_shape=tuple(spatial_shape),
            output_spatial_shape=(out_D, out_H, out_W),
            kernel_size=(kD, kH, kW),
            stride=(sz, sy, sx),
            padding=(pz, py, px),
            dilation=(dz, dy, dx),
            subm=False,
        )

    cat_in = torch.cat(all_in_ids)
    cat_out = torch.cat(all_out_coords, dim=0)
    cat_k = torch.cat(all_k_ids)

    encoded = cat_out[:, 0] * (out_D * out_H * out_W) + cat_out[:, 1] * (out_H * out_W) + cat_out[:, 2] * out_W + cat_out[:, 3]
    unique_enc, inverse = torch.unique(encoded, sorted=True, return_inverse=True)

    # Decode unique output coords
    ub = unique_enc // (out_D * out_H * out_W)
    rem = unique_enc % (out_D * out_H * out_W)
    uz = rem // (out_H * out_W)
    rem = rem % (out_H * out_W)
    uy = rem // out_W
    ux = rem % out_W
    out_indices = torch.stack([ub, uz, uy, ux], dim=1).to(torch.int32)

    pairs_in: List[torch.Tensor] = []
    pairs_out: List[torch.Tensor] = []
    pair_num: List[int] = []

    for k in range(Kvol):
        mask = cat_k == k
        if mask.any():
            pairs_in.append(cat_in[mask])
            pairs_out.append(inverse[mask])
            pair_num.append(mask.sum().item())
        else:
            pairs_in.append(torch.empty(0, dtype=torch.long, device=device))
            pairs_out.append(torch.empty(0, dtype=torch.long, device=device))
            pair_num.append(0)

    return IndiceData(
        out_indices=out_indices,
        pairs_in=pairs_in,
        pairs_out=pairs_out,
        pair_num=torch.tensor(pair_num, dtype=torch.long),
        input_indices=indices,
        input_spatial_shape=tuple(spatial_shape),
        output_spatial_shape=(out_D, out_H, out_W),
        kernel_size=(kD, kH, kW),
        stride=(sz, sy, sx),
        padding=(pz, py, px),
        dilation=(dz, dy, dx),
        subm=False,
    )
