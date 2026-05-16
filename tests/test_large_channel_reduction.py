"""Regression test for NaN in SubMConv3d with large channel-reduction.

Bug: SubMConv3d(in_channels=2048, out_channels=128, kernel_size=3) produces NaN
output from finite input when operating on ~8000 voxels.

Observed in the AniGen SLAT flow model's SparseResBlock3d upsample path:
  - SparseUpsample(2) expands 1948 → 8083 voxels (128→2048 ch via upstream block)
  - LayerNorm32 + SiLU applied to 2048-ch features
  - SubMConv3d(2048, 128, 3) first introduces NaN

The reproducing conditions are:
  - C_in=2048, C_out=128, k=3 (large channel reduction)
  - ~8000 active voxels, batch_size=1
  - Input values in [-4000, 4000] (pre-LayerNorm range seen in practice)
  - Input values in [-4, 4]    (post-LayerNorm/SiLU range, still reproduces)
"""
import pytest
import torch
import numpy as np

import trispconv.pytorch as spconv
from trispconv.testing.sparse_data import generate_sparse_data, sparse_data_to_tensors


SPATIAL_SHAPE = [32, 32, 32]  # big enough for ~8000 voxels at ~25% density


def _make_subm_input(n_voxels, C_in, spatial_shape, seed, device="cuda"):
    """Return (features, indices) for a SubMConv test."""
    data = generate_sparse_data(spatial_shape, [n_voxels], C_in, seed=seed)
    features, _, indices = sparse_data_to_tensors(data)
    return features.to(device), indices.to(device)


def _run_subm(C_in, C_out, k, features, indices, spatial_shape, seed, dtype=torch.float32):
    """Construct a SubMConv3d, load random weights, run forward, return output features."""
    batch_size = 1
    conv = spconv.SubMConv3d(C_in, C_out, k, bias=False)
    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.1, 0.1, size=(C_out, k, k, k, C_in)).astype(np.float32)
    conv.weight.data.copy_(torch.from_numpy(w))
    conv = conv.to(features.device).to(dtype)

    x = spconv.SparseConvTensor(features.to(dtype), indices, spatial_shape, batch_size)
    y = conv(x)
    return y.features


# ---------------------------------------------------------------------------
# Primary regression: the exact config that triggered the bug
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_subm_2048_to_128_no_nan(dtype):
    """SubMConv3d(2048→128, k=3) must not produce NaN from finite input.

    fp16 is the dtype used by AniGen's SLAT flow model weights and is where
    the 2048×27=55296 MAC accumulation overflow was first observed.
    """
    C_in, C_out, k = 2048, 128, 3
    features, indices = _make_subm_input(8083, C_in, SPATIAL_SHAPE, seed=42)

    # Use the large pre-norm range seen in practice (values in [-4000, 4000])
    features = features * 4000.0

    out = _run_subm(C_in, C_out, k, features, indices, SPATIAL_SHAPE, seed=0, dtype=dtype)

    assert not out.isnan().any(), (
        f"SubMConv3d({C_in}→{C_out}, k={k}) [{dtype}]: NaN in output for pre-norm "
        f"input range (in min={features.min():.1f} max={features.max():.1f})"
    )
    # Inf is acceptable here: per-slot sum ≈ C_in × |feat| × |w| ≈ 2048×4000×0.1 ≈ 819K
    # which exceeds fp16 max (~65504). spconv also produces ±Inf for out-of-range inputs;
    # the bug was NaN (Inf−Inf from fp16 accumulator overflow), not Inf per se.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_subm_2048_to_128_post_norm_no_nan(dtype):
    """SubMConv3d(2048→128, k=3) must not produce NaN from post-LayerNorm input."""
    C_in, C_out, k = 2048, 128, 3
    features, indices = _make_subm_input(8083, C_in, SPATIAL_SHAPE, seed=42)

    # Simulate post-LayerNorm + SiLU values: unit-scale, non-negative
    features = torch.nn.functional.silu(features)

    out = _run_subm(C_in, C_out, k, features, indices, SPATIAL_SHAPE, seed=0, dtype=dtype)

    assert not out.isnan().any(), (
        f"SubMConv3d({C_in}→{C_out}, k={k}) [{dtype}]: NaN in output for post-norm input"
    )


# ---------------------------------------------------------------------------
# Parametrized sweep over large channel-reduction configs
# ---------------------------------------------------------------------------

LARGE_REDUCTION_CASES = [
    pytest.param(C_in, C_out, k, n, dtype,
                 id=f"C{C_in}-{C_out}-k{k}-n{n}-{dtype_name}")
    for C_in, C_out in [(2048, 128), (2048, 256), (1024, 128), (1024, 64)]
    for k in [1, 3]
    for n in [1000, 8000]
    for dtype, dtype_name in [(torch.float32, "fp32"), (torch.float16, "fp16")]
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("C_in,C_out,k,n_voxels,dtype", LARGE_REDUCTION_CASES)
def test_subm_large_reduction_no_nan(C_in, C_out, k, n_voxels, dtype):
    """SubMConv3d with large channel reductions must never produce NaN."""
    features, indices = _make_subm_input(n_voxels, C_in, SPATIAL_SHAPE,
                                          seed=hash((C_in, C_out, k, n_voxels)) % (2**31))
    features = features * 100.0  # amplify to catch overflow-class bugs

    out = _run_subm(C_in, C_out, k, features, indices, SPATIAL_SHAPE,
                    seed=hash((C_in, C_out, k)) % (2**31), dtype=dtype)

    assert not out.isnan().any(), (
        f"SubMConv3d({C_in}→{C_out}, k={k}, n={n_voxels}) [{dtype}]: NaN in output"
    )
    assert not out.isinf().any(), (
        f"SubMConv3d({C_in}→{C_out}, k={k}, n={n_voxels}) [{dtype}]: Inf in output"
    )
