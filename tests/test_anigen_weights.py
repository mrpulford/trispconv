"""Weight compatibility test against real AniGen checkpoint tensors.

Does not require AniGen to be importable. Loads the actual checkpoint,
identifies all sparse conv weights by shape, constructs matching trispconv
modules, and verifies clean load + normalization.

Skipped automatically when the checkpoint is not present (CI / fresh clone).
"""
import pytest
import torch
from pathlib import Path

import trispconv.pytorch as spconv
from trispconv.pytorch.weight import normalize_spconv_weight

CKPT_PATH = Path(__file__).parent.parent / "local/repos/AniGen/ckpts/anigen/slat_dae/ckpts/decoder_final.pt"


def pytest_configure(config):
    config.addinivalue_line("markers", "anigen: requires AniGen checkpoint")


@pytest.fixture(scope="module")
def decoder_state_dict():
    if not CKPT_PATH.exists():
        pytest.skip("AniGen checkpoint not found")
    return torch.load(CKPT_PATH, map_location="cpu", weights_only=True)


def _parse_krsc(shape):
    """Return (C_out, kD, kH, kW, C_in) from a 5-tuple, or None if not 5D KRSC."""
    if len(shape) != 5:
        return None
    C_out, kD, kH, kW, C_in = shape
    if kD == kH == kW:
        return C_out, kD, kH, kW, C_in
    return None


# ---------------------------------------------------------------------------
# All 5D weights from the checkpoint must normalize cleanly
# ---------------------------------------------------------------------------

def test_all_sparse_weights_normalize(decoder_state_dict):
    """normalize_spconv_weight must succeed on every 5D tensor in the checkpoint."""
    sd = decoder_state_dict
    found = 0
    for key, tensor in sd.items():
        parsed = _parse_krsc(tensor.shape)
        if parsed is None:
            continue
        C_out, kD, kH, kW, C_in = parsed
        w_internal = normalize_spconv_weight(tensor, (kD, kH, kW), C_in, C_out)
        Kvol = kD * kH * kW
        assert w_internal.shape == (Kvol, C_in, C_out), \
            f"{key}: expected ({Kvol}, {C_in}, {C_out}), got {w_internal.shape}"
        found += 1
    assert found > 0, "No 5D sparse conv weights found in checkpoint"
    print(f"\n  Normalized {found} sparse conv weight tensors successfully")


# ---------------------------------------------------------------------------
# Construct matching trispconv modules and do load_state_dict
# ---------------------------------------------------------------------------

def test_upsample_block_weight_load(decoder_state_dict):
    """Construct SubM/InverseConv modules matching upsample block 0 and load weights."""
    sd = decoder_state_dict

    # upsample.0 uses:
    #   out_layers.0.conv: SparseInverseConv3d(768, 192, 3)
    #   out_layers.3.conv: SubMConv3d(192, 192, 3)
    #   skip_connection.conv: SparseInverseConv3d(768, 192, 1)
    w_inv3  = sd["upsample.0.out_layers.0.conv.weight"]   # (192, 3, 3, 3, 768)
    w_subm3 = sd["upsample.0.out_layers.3.conv.weight"]   # (192, 3, 3, 3, 192)
    w_inv1  = sd["upsample.0.skip_connection.conv.weight"] # (192, 1, 1, 1, 768)

    assert w_inv3.shape  == (192, 3, 3, 3, 768)
    assert w_subm3.shape == (192, 3, 3, 3, 192)
    assert w_inv1.shape  == (192, 1, 1, 1, 768)

    # Construct modules
    inv3  = spconv.SparseInverseConv3d(768, 192, 3, bias=False, indice_key="up0")
    subm3 = spconv.SubMConv3d(192, 192, 3, bias=False)
    inv1  = spconv.SparseInverseConv3d(768, 192, 1, bias=False, indice_key="up0_skip")

    # Load weights directly
    inv3.weight.data.copy_(w_inv3)
    subm3.weight.data.copy_(w_subm3)
    inv1.weight.data.copy_(w_inv1)

    # Verify normalization round-trips
    assert normalize_spconv_weight(inv3.weight, (3,3,3), 768, 192).shape == (27, 768, 192)
    assert normalize_spconv_weight(subm3.weight, (3,3,3), 192, 192).shape == (27, 192, 192)
    assert normalize_spconv_weight(inv1.weight, (1,1,1), 768, 192).shape == (1, 768, 192)


def test_all_unique_sparse_shapes(decoder_state_dict):
    """Report all unique (C_out, k, C_in) combos — useful for auditing coverage."""
    sd = decoder_state_dict
    shapes = set()
    for key, tensor in sd.items():
        if tensor.ndim == 5:
            C_out, kD, kH, kW, C_in = tensor.shape
            shapes.add((C_in, C_out, kD))
    assert len(shapes) > 0
    # Print for visibility in -v mode
    for C_in, C_out, k in sorted(shapes):
        print(f"\n  SparseConv: C_in={C_in:4d}  C_out={C_out:4d}  k={k}x{k}x{k}")
