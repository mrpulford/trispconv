"""SubM indice pair generation tests, validated against dense PyTorch conv3d."""
import torch
import pytest
import trispconv.pytorch as spconv
from trispconv.pytorch.indices import generate_subm_indice_pairs


def _dense_subm_reference(features, indices, spatial_shape, weight, bias, batch_size):
    """Run SubM conv via dense PyTorch conv3d and re-extract sparse output."""
    N, C_in = features.shape
    C_out = weight.shape[0]
    kD, kH, kW = weight.shape[1], weight.shape[2], weight.shape[3]
    D, H, W = spatial_shape

    # Scatter sparse input into dense
    dense_in = torch.zeros(batch_size, C_in, D, H, W, dtype=features.dtype)
    idx = indices.long()
    dense_in[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]] = features

    # weight is KRSC (C_out, kD, kH, kW, C_in) -> need (C_out, C_in, kD, kH, kW) for conv3d
    w_conv = weight.permute(0, 4, 1, 2, 3).contiguous()
    pad = (kD // 2, kH // 2, kW // 2)
    dense_out = torch.nn.functional.conv3d(dense_in, w_conv, bias=bias, padding=pad)

    # Re-extract at input coordinates (SubM: same active set)
    out_feats = dense_out[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]
    return out_feats


def test_subm_3x3x3_vs_dense():
    torch.manual_seed(0)
    B, N, C_in, C_out = 1, 12, 4, 8
    D, H, W = 10, 10, 10

    coords = torch.randint(1, 9, (N, 3), dtype=torch.int32)
    indices = torch.cat([torch.zeros(N, 1, dtype=torch.int32), coords], dim=1)
    features = torch.randn(N, C_in)

    # KRSC weight: (C_out, kD, kH, kW, C_in)
    weight_krsc = torch.randn(C_out, 3, 3, 3, C_in)
    bias = torch.randn(C_out)

    # Reference
    ref = _dense_subm_reference(features, indices, (D, H, W), weight_krsc, bias, B)

    # trispconv SubMConv3d
    conv = spconv.SubMConv3d(C_in, C_out, 3, bias=True)
    conv.weight.data.copy_(weight_krsc)
    conv.bias.data.copy_(bias)

    x = spconv.SparseConvTensor(features, indices, [D, H, W], B)
    y = conv(x)

    assert y.features.shape == (N, C_out)
    assert torch.allclose(y.features, ref, atol=1e-4, rtol=1e-4), \
        f"max diff: {(y.features - ref).abs().max()}"


def test_subm_1x1x1_vs_dense():
    torch.manual_seed(1)
    B, N, C_in, C_out = 1, 8, 6, 6
    D, H, W = 8, 8, 8

    coords = torch.randint(0, 8, (N, 3), dtype=torch.int32)
    indices = torch.cat([torch.zeros(N, 1, dtype=torch.int32), coords], dim=1)
    features = torch.randn(N, C_in)

    weight_krsc = torch.randn(C_out, 1, 1, 1, C_in)

    # spconv k=1 uses features @ weight.view(C_in, C_out), NOT the standard W.T path
    w2d = weight_krsc.reshape(C_out * C_in).view(C_in, C_out)
    ref = features @ w2d

    conv = spconv.SubMConv3d(C_in, C_out, 1, bias=False)
    conv.weight.data.copy_(weight_krsc)

    x = spconv.SparseConvTensor(features, indices, [D, H, W], B)
    y = conv(x)

    assert torch.allclose(y.features, ref, atol=1e-4, rtol=1e-4)


def test_subm_preserves_active_set():
    torch.manual_seed(2)
    N, C_in, C_out = 15, 4, 8
    D, H, W = 12, 12, 12

    coords = torch.randint(1, 11, (N, 3), dtype=torch.int32)
    indices = torch.cat([torch.zeros(N, 1, dtype=torch.int32), coords], dim=1)
    features = torch.randn(N, C_in)

    conv = spconv.SubMConv3d(C_in, C_out, 3, bias=False)
    x = spconv.SparseConvTensor(features, indices, [D, H, W], 1)
    y = conv(x)

    assert torch.equal(y.indices, x.indices), "SubM must preserve active coordinate set"
    assert y.spatial_shape == x.spatial_shape


def test_subm_indice_key_caching():
    torch.manual_seed(3)
    N, C = 10, 4
    D, H, W = 8, 8, 8

    coords = torch.randint(1, 7, (N, 3), dtype=torch.int32)
    indices = torch.cat([torch.zeros(N, 1, dtype=torch.int32), coords], dim=1)
    features = torch.randn(N, C)

    conv = spconv.SubMConv3d(C, C, 3, bias=False, indice_key="subm0")
    x = spconv.SparseConvTensor(features, indices, [D, H, W], 1)
    y = conv(x)

    assert "subm0" in y.indice_dict
    # Second pass reuses cached indice data
    y2 = conv(y)
    assert "subm0" in y2.indice_dict


def test_subm_batch_size_2():
    torch.manual_seed(4)
    N, C_in, C_out = 20, 4, 6
    D, H, W = 8, 8, 8

    coords = torch.randint(1, 7, (N, 3), dtype=torch.int32)
    batch = torch.cat([torch.zeros(N // 2, 1, dtype=torch.int32),
                       torch.ones(N // 2, 1, dtype=torch.int32)], dim=0)
    indices = torch.cat([batch, coords], dim=1)
    features = torch.randn(N, C_in)

    weight_krsc = torch.randn(C_out, 3, 3, 3, C_in)
    conv = spconv.SubMConv3d(C_in, C_out, 3, bias=False)
    conv.weight.data.copy_(weight_krsc)

    x = spconv.SparseConvTensor(features, indices, [D, H, W], 2)
    y = conv(x)
    assert y.features.shape == (N, C_out)
    assert y.batch_size == 2
