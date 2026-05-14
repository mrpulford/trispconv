"""Strided sparse conv and inverse conv tests, validated against dense PyTorch conv3d."""
import torch
import pytest
import trispconv.pytorch as spconv


def _scatter_to_dense(features, indices, spatial_shape, batch_size, C):
    D, H, W = spatial_shape
    dense = torch.zeros(batch_size, C, D, H, W, dtype=features.dtype)
    idx = indices.long()
    dense[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]] = features
    return dense


def _dense_strided_reference(features, indices, spatial_shape, weight_krsc, bias, batch_size, stride, padding):
    C_in = features.shape[1]
    C_out, kD, kH, kW, _ = weight_krsc.shape
    D, H, W = spatial_shape

    dense_in = _scatter_to_dense(features, indices, spatial_shape, batch_size, C_in)
    w_conv = weight_krsc.permute(0, 4, 1, 2, 3).contiguous()
    dense_out = torch.nn.functional.conv3d(dense_in, w_conv, bias=bias, stride=stride, padding=padding)
    return dense_out


def _extract_sparse(dense_out, out_indices):
    idx = out_indices.long()
    return dense_out[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]


def test_strided_2x2x2_vs_dense():
    torch.manual_seed(10)
    B, C_in, C_out = 1, 4, 8
    D, H, W = 10, 10, 10
    stride, padding, ksize = 2, 1, 3

    # Dense input (all active for easy reference comparison)
    all_coords = []
    for z in range(D):
        for y in range(H):
            for x in range(W):
                all_coords.append([0, z, y, x])
    indices = torch.tensor(all_coords, dtype=torch.int32)
    features = torch.randn(len(indices), C_in)

    weight_krsc = torch.randn(C_out, ksize, ksize, ksize, C_in)
    bias = torch.randn(C_out)

    conv = spconv.SparseConv3d(C_in, C_out, ksize, stride=stride, padding=padding, bias=True)
    conv.weight.data.copy_(weight_krsc)
    conv.bias.data.copy_(bias)

    x = spconv.SparseConvTensor(features, indices, [D, H, W], B)
    y = conv(x)

    # Dense reference
    dense_out = _dense_strided_reference(features, indices, (D, H, W), weight_krsc, bias, B, stride, padding)
    ref = _extract_sparse(dense_out, y.indices)

    assert torch.allclose(y.features, ref, atol=1e-4, rtol=1e-4), \
        f"max diff: {(y.features - ref).abs().max()}"


def test_strided_output_shape():
    torch.manual_seed(11)
    B, C_in, C_out = 1, 4, 8
    D, H, W = 8, 8, 8

    coords = torch.randint(0, 8, (30, 3), dtype=torch.int32)
    indices = torch.cat([torch.zeros(30, 1, dtype=torch.int32), coords], dim=1)
    features = torch.randn(30, C_in)

    conv = spconv.SparseConv3d(C_in, C_out, 3, stride=2, padding=1, bias=False)
    x = spconv.SparseConvTensor(features, indices, [D, H, W], B)
    y = conv(x)

    # Formula: floor((8 + 2*1 - 1*(3-1) - 1) / 2 + 1) = floor(7/2+1) = 4
    assert y.spatial_shape == [4, 4, 4]
    assert y.indices[:, 1:].max() < 4


def test_strided_indice_key_stored():
    torch.manual_seed(12)
    B, N, C_in, C_out = 1, 20, 4, 8
    D, H, W = 8, 8, 8

    coords = torch.randint(0, 8, (N, 3), dtype=torch.int32)
    indices = torch.cat([torch.zeros(N, 1, dtype=torch.int32), coords], dim=1)
    features = torch.randn(N, C_in)

    conv = spconv.SparseConv3d(C_in, C_out, 3, stride=2, padding=1, bias=False, indice_key="down1")
    x = spconv.SparseConvTensor(features, indices, [D, H, W], B)
    y = conv(x)

    assert "down1" in y.indice_dict


def test_inverse_conv_restores_coords():
    torch.manual_seed(13)
    B, N, C_in, C_mid, C_out = 1, 20, 4, 8, 4
    D, H, W = 8, 8, 8

    coords = torch.randint(0, 8, (N, 3), dtype=torch.int32)
    indices = torch.cat([torch.zeros(N, 1, dtype=torch.int32), coords], dim=1)
    features = torch.randn(N, C_in)

    down = spconv.SparseConv3d(C_in, C_mid, 3, stride=2, padding=1, bias=False, indice_key="down1")
    up = spconv.SparseInverseConv3d(C_mid, C_out, 3, bias=False, indice_key="down1")

    x = spconv.SparseConvTensor(features, indices, [D, H, W], B)
    y = down(x)
    z = up(y)

    assert torch.equal(z.indices, x.indices), "Inverse conv must restore original coordinates"
    assert z.spatial_shape == [D, H, W]
    assert z.features.shape == (N, C_out)


def test_inverse_conv_missing_key_raises():
    conv = spconv.SparseInverseConv3d(4, 4, 3, indice_key="missing")
    x = spconv.SparseConvTensor(
        torch.randn(5, 4),
        torch.zeros(5, 4, dtype=torch.int32),
        [8, 8, 8], 1,
    )
    with pytest.raises(KeyError):
        conv(x)
