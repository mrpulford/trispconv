import torch
import pytest
import trispconv.pytorch as spconv


def make_sparse(B=1, N=8, C=4, D=8, H=8, W=8):
    coords = torch.randint(0, min(D, H, W), (N, 3), dtype=torch.int32)
    batch_col = torch.zeros(N, 1, dtype=torch.int32)
    indices = torch.cat([batch_col, coords], dim=1)
    features = torch.randn(N, C)
    return spconv.SparseConvTensor(features, indices, [D, H, W], B)


def test_construction():
    x = make_sparse()
    assert x.features.shape == (8, 4)
    assert x.indices.shape == (8, 4)
    assert x.spatial_shape == [8, 8, 8]
    assert x.batch_size == 1


def test_replace_feature():
    x = make_sparse()
    new_feats = torch.ones(8, 4)
    y = x.replace_feature(new_feats)
    assert torch.equal(y.features, new_feats)
    assert y.indices is x.indices
    # original unchanged
    assert not torch.equal(x.features, new_feats)


def test_features_setter_raises():
    x = make_sparse()
    with pytest.raises(AttributeError):
        x.features = torch.zeros(8, 4)


def test_dense_shape():
    x = make_sparse(B=2, N=10, C=3, D=4, H=5, W=6)
    # Give each voxel a different batch
    indices = x.indices.clone()
    indices[:5, 0] = 0
    indices[5:, 0] = 1
    x2 = spconv.SparseConvTensor(x.features, indices, [4, 5, 6], 2)
    dense = x2.dense()
    assert dense.shape == (2, 3, 4, 5, 6)


def test_find_indice_pair_missing():
    x = make_sparse()
    assert x.find_indice_pair("nonexistent") is None


def test_find_indice_pair_none_key():
    x = make_sparse()
    assert x.find_indice_pair(None) is None
