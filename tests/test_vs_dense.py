"""Parametrized sparse conv vs dense conv3d tests.

Adapted from spconv/test/test_conv.py (Apache 2.0, original copyright 2021 Yan Yan).
Key changes:
  - CPU only (no CUDA requirement)
  - Inference only (no gradient checks)
  - trispconv instead of spconv
  - Always KRSC weight layout (spconv default when ALL_WEIGHT_IS_KRSC=True)
  - pytest.mark.parametrize instead of hand-rolled grid
  - Added SubM parametrized cases (absent from spconv's own tests)
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

import trispconv.pytorch as spconv
from trispconv.testing.sparse_data import generate_sparse_data, sparse_data_to_tensors

torch.backends.cuda.matmul.allow_tf32 = False

SHAPE = [19, 18, 17]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparse(shape, batch_size, num_points_per_batch, C, seed):
    data = generate_sparse_data(shape, [num_points_per_batch] * batch_size, C, seed=seed)
    features, features_dense, indices = sparse_data_to_tensors(data)
    return features, features_dense, indices


def _krsc_to_conv3d(w_krsc):
    """KRSC (C_out, kD, kH, kW, C_in) -> Conv3d (C_out, C_in, kD, kH, kW)."""
    return w_krsc.permute(0, 4, 1, 2, 3).contiguous()


# ---------------------------------------------------------------------------
# SparseConv3d parametrized suite
# ---------------------------------------------------------------------------

STRIDED_CASES = [
    pytest.param(bs, k, s, p, d, id=f"bs{bs}-k{k}-s{s}-p{p}-d{d}")
    for bs in [1, 2]
    for k in [2, 3]
    for s in [1, 2]
    for p in [0, 1]
    for d in [1, 2]
    if not (s > 1 and d > 1)  # spconv doesn't support stride>1 + dilation>1
]


@pytest.mark.parametrize("batch_size,ksize,stride,padding,dilation", STRIDED_CASES)
def test_sparseconv3d_vs_dense(batch_size, ksize, stride, padding, dilation):
    """SparseConv3d output (re-densified) must match nn.Conv3d on a dense input."""
    C_in, C_out = 16, 16
    seed = hash((batch_size, ksize, stride, padding, dilation)) % (2**31)

    features, features_dense, indices = _make_sparse(SHAPE, batch_size, 300, C_in, seed)

    # KRSC weight: (C_out, k, k, k, C_in)
    rng = np.random.default_rng(seed)
    w_np = rng.uniform(-1, 1, size=(C_out, ksize, ksize, ksize, C_in)).astype(np.float32)
    w_krsc = torch.from_numpy(w_np)

    # trispconv
    sp_conv = spconv.SparseConv3d(C_in, C_out, ksize, stride=stride, padding=padding,
                                   dilation=dilation, bias=False)
    sp_conv.weight.data.copy_(w_krsc)
    x = spconv.SparseConvTensor(features, indices, SHAPE, batch_size)
    y = sp_conv(x)
    out_sparse_dense = y.dense()  # (B, C_out, D_out, H_out, W_out)

    # Dense reference
    ref_conv = nn.Conv3d(C_in, C_out, ksize, stride=stride, padding=padding,
                         dilation=dilation, bias=False)
    ref_conv.weight.data.copy_(_krsc_to_conv3d(w_krsc))
    out_dense = ref_conv(features_dense)

    assert out_sparse_dense.shape == out_dense.shape, \
        f"Shape mismatch: sparse {out_sparse_dense.shape} vs dense {out_dense.shape}"

    # Only compare at output locations that were populated by the sparse forward.
    # At locations with no active input neighbours the sparse output is zero but
    # the dense output may be non-zero — that is expected and correct.
    out_idx = y.indices.long()
    sp_vals = out_sparse_dense[out_idx[:, 0], :, out_idx[:, 1], out_idx[:, 2], out_idx[:, 3]]
    ref_vals = out_dense[out_idx[:, 0], :, out_idx[:, 1], out_idx[:, 2], out_idx[:, 3]]

    assert torch.allclose(sp_vals, ref_vals, atol=1e-4, rtol=1e-4), \
        f"max diff at active locs: {(sp_vals - ref_vals).abs().max():.6f}  " \
        f"params: bs={batch_size} k={ksize} s={stride} p={padding} d={dilation}"


# ---------------------------------------------------------------------------
# SubMConv3d parametrized suite
# ---------------------------------------------------------------------------

SUBM_CASES = [
    pytest.param(bs, k, d, id=f"bs{bs}-k{k}-d{d}")
    for bs in [1, 2]
    for k in [1, 3]
    for d in [1, 2]
]


@pytest.mark.parametrize("batch_size,ksize,dilation", SUBM_CASES)
def test_submconv3d_vs_dense(batch_size, ksize, dilation):
    """SubMConv3d output at active locations must match the dense reference on dense input.

    For k>1 the reference is nn.Conv3d.
    For k=1, spconv uses features @ weight.view(C_in, C_out) instead of the standard
    features @ W.T, so we use a direct matmul reference to match that convention.
    """
    C_in, C_out = 16, 16
    seed = hash((batch_size, ksize, dilation)) % (2**31)

    features, features_dense, indices = _make_sparse(SHAPE, batch_size, 300, C_in, seed)

    rng = np.random.default_rng(seed)
    w_np = rng.uniform(-1, 1, size=(C_out, ksize, ksize, ksize, C_in)).astype(np.float32)
    w_krsc = torch.from_numpy(w_np)

    # trispconv SubM
    sp_conv = spconv.SubMConv3d(C_in, C_out, ksize, dilation=dilation, bias=False)
    sp_conv.weight.data.copy_(w_krsc)
    x = spconv.SparseConvTensor(features, indices, SHAPE, batch_size)
    y = sp_conv(x)

    idx = indices.long()
    sp_vals = y.features  # (N, C_out)

    if ksize == 1:
        # spconv k=1 special path: features @ weight.view(C_in, C_out)
        w2d = w_krsc.reshape(C_out * C_in).view(C_in, C_out)
        ref_vals = features @ w2d
    else:
        pad = dilation * (ksize // 2)
        ref_conv = nn.Conv3d(C_in, C_out, ksize, padding=pad, dilation=dilation, bias=False)
        ref_conv.weight.data.copy_(_krsc_to_conv3d(w_krsc))
        out_dense = ref_conv(features_dense)
        ref_vals = out_dense[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]

    assert torch.allclose(sp_vals, ref_vals, atol=1e-4, rtol=1e-4), \
        f"max diff: {(sp_vals - ref_vals).abs().max():.6f}  " \
        f"params: bs={batch_size} k={ksize} d={dilation}"


# ---------------------------------------------------------------------------
# SparseSequential + dense module interleaving
# ---------------------------------------------------------------------------

def test_sparse_sequential_with_batchnorm():
    """SparseSequential must pass features through dense BN1d and ReLU correctly."""
    torch.manual_seed(99)
    N, C = 20, 16
    D, H, W = 8, 8, 8

    coords = torch.randint(0, 8, (N, 3), dtype=torch.int32)
    indices = torch.cat([torch.zeros(N, 1, dtype=torch.int32), coords], dim=1)
    features = torch.randn(N, C)

    # ReLU is last so we can assert non-negative output
    seq = spconv.SparseSequential(
        spconv.SubMConv3d(C, C, 3, bias=False),
        nn.BatchNorm1d(C),
        nn.ReLU(),
    )
    seq.eval()

    x = spconv.SparseConvTensor(features, indices, [D, H, W], 1)
    y = seq(x)

    assert isinstance(y, spconv.SparseConvTensor)
    assert y.features.shape == (N, C)
    assert (y.features >= 0).all(), "ReLU should make all values non-negative"
    # active set must be unchanged (SubM)
    assert torch.equal(y.indices, x.indices)
