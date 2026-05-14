"""Sparse test data generation, adapted from spconv/test_utils.py (Apache 2.0).

Original copyright 2021 Yan Yan.
"""
from __future__ import annotations

from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import torch


def generate_sparse_data(
    shape: List[int],
    num_points: List[int],
    num_channels: int,
    data_range: Tuple[float, float] = (-1.0, 1.0),
    dtype=np.float32,
    seed: Optional[int] = None,
):
    """Generate random sparse data and a matching dense tensor.

    Returns a dict with keys:
      features       (N_total, C)  float32
      features_dense (B, C, *shape) float32
      indices        (N_total, ndim+1) int32  — last column is batch index
    """
    rng = np.random.default_rng(seed)
    ndim = len(shape)
    num_points = np.array(num_points)
    batch_size = len(num_points)

    all_coords = np.stack(
        np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1
    ).reshape(-1, ndim)

    batch_indices = []
    for i in range(batch_size):
        perm = rng.permutation(len(all_coords))[: num_points[i]]
        inds = all_coords[perm]
        inds = np.pad(inds, ((0, 0), (0, 1)), constant_values=i)  # append batch col
        batch_indices.append(inds)

    sparse_data = rng.uniform(
        data_range[0], data_range[1], size=(num_points.sum(), num_channels)
    ).astype(dtype)

    dense_data = np.zeros([batch_size, num_channels, *shape], dtype=dtype)
    start = 0
    for i, inds in enumerate(batch_indices):
        for j, ind in enumerate(inds):
            dense_data[(i, slice(None), *ind[:-1])] = sparse_data[start + j]
        start += len(inds)

    indices = np.concatenate(batch_indices, axis=0).astype(np.int32)

    return {
        "features": sparse_data,
        "features_dense": dense_data,
        "indices": indices,  # (N, ndim+1), last col = batch
    }


def sparse_data_to_tensors(sparse_dict, device="cpu"):
    """Convert generate_sparse_data output to torch tensors.

    spconv convention: indices are (N, 4) with batch in column 0, spatial in 1-3.
    generate_sparse_data puts batch in the last column, so we reorder.
    """
    features = torch.from_numpy(sparse_dict["features"]).to(device)
    features_dense = torch.from_numpy(sparse_dict["features_dense"]).to(device)
    raw_indices = sparse_dict["indices"]  # (N, ndim+1), spatial cols first, batch last
    # reorder to [batch, z, y, x]
    indices = torch.from_numpy(
        np.ascontiguousarray(raw_indices[:, [-1, 0, 1, 2]])
    ).int().to(device)
    return features, features_dense, indices
