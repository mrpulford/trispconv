from __future__ import annotations

from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from trispconv.pytorch.indices import IndiceData


class SparseConvTensor:
    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Sequence[int],
        batch_size: int,
        grid=None,
        voxel_num=None,
        indice_dict: Optional[Dict] = None,
        benchmark: bool = False,
    ):
        assert features.ndim == 2, f"features must be (N, C), got {features.shape}"
        assert indices.ndim == 2, f"indices must be (N, ndim+1), got {indices.shape}"
        assert indices.dtype == torch.int32, "indices must be int32"

        self._features = features
        self.indices = indices
        self.spatial_shape = list(spatial_shape)
        self.batch_size = batch_size
        self.indice_dict: Dict[str, "IndiceData"] = indice_dict if indice_dict is not None else {}
        self.grid = grid
        self.voxel_num = voxel_num
        self.benchmark = benchmark
        self.benchmark_record: dict = {}

    def __repr__(self) -> str:
        return f"SparseConvTensor[features={self._features.shape}, spatial={self.spatial_shape}]"

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @features.setter
    def features(self, val: torch.Tensor):
        raise AttributeError(
            "Use x = x.replace_feature(new_features) instead of assigning x.features directly."
        )

    def replace_feature(self, features: torch.Tensor) -> "SparseConvTensor":
        out = SparseConvTensor(
            features,
            self.indices,
            self.spatial_shape,
            self.batch_size,
            self.grid,
            self.voxel_num,
            self.indice_dict,
            self.benchmark,
        )
        out.benchmark_record = self.benchmark_record
        return out

    def shadow_copy(self) -> "SparseConvTensor":
        return self.replace_feature(self._features)

    def find_indice_pair(self, key) -> Optional["IndiceData"]:
        if key is None:
            return None
        return self.indice_dict.get(key, None)

    def dense(self, channels_first: bool = True) -> torch.Tensor:
        """Scatter sparse features into a dense tensor.

        Returns shape (B, C, D, H, W) when channels_first=True.
        """
        B = self.batch_size
        C = self._features.shape[1]
        D, H, W = self.spatial_shape

        dense = torch.zeros(
            (B, D, H, W, C),
            dtype=self._features.dtype,
            device=self._features.device,
        )

        idx = self.indices.long()
        b, z, y, x = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]

        assert (b >= 0).all() and (b < B).all(), "batch index out of range"
        assert (z >= 0).all() and (z < D).all(), "z coordinate out of range"
        assert (y >= 0).all() and (y < H).all(), "y coordinate out of range"
        assert (x >= 0).all() and (x < W).all(), "x coordinate out of range"

        dense[b, z, y, x] = self._features

        if not channels_first:
            return dense  # (B, D, H, W, C)
        return dense.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
