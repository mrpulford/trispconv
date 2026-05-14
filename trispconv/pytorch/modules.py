from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from trispconv.pytorch.tensor import SparseConvTensor
from trispconv.pytorch.indices import (
    IndiceData,
    generate_subm_indice_pairs,
    generate_strided_indice_pairs,
    _to3,
    _output_spatial_shape,
)
from trispconv.pytorch.weight import normalize_spconv_weight
from trispconv.pytorch.functional import sparse_conv_forward


class SparseModule(nn.Module):
    """Marker base class for modules that consume and produce SparseConvTensor."""
    pass


class SparseConvolution(SparseModule):
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        subm: bool = False,
        output_padding=0,
        transposed: bool = False,
        inverse: bool = False,
        indice_key: Optional[str] = None,
        algo=None,
    ):
        super().__init__()

        if ndim != 3:
            raise NotImplementedError(f"Only ndim=3 is supported, got ndim={ndim}")
        if groups != 1:
            raise NotImplementedError("Grouped convolution is not supported")
        if transposed and not inverse:
            raise NotImplementedError("Transposed convolution is not supported; use SparseInverseConv3d")

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to3(kernel_size)
        self.stride = _to3(stride)
        self.padding = _to3(padding)
        self.dilation = _to3(dilation)
        self.subm = subm
        self.inverse = inverse
        self.indice_key = indice_key

        kD, kH, kW = self.kernel_size
        Kvol = kD * kH * kW

        # Internal weight layout: (Kvol, C_in, C_out)
        # Initialized in KRSC layout (spconv convention) for checkpoint compatibility;
        # normalized on first forward or explicit load.
        self.weight = nn.Parameter(
            torch.empty(out_channels, kD, kH, kW, in_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight.data.reshape(out_channels, Kvol * in_channels))

    def _get_weight(self) -> torch.Tensor:
        return normalize_spconv_weight(
            self.weight,
            self.kernel_size,
            self.in_channels,
            self.out_channels,
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        if self.inverse:
            return self._forward_inverse(x)
        if self.subm:
            return self._forward_subm(x)
        return self._forward_strided(x)

    def _get_or_compute_subm_indice(self, x: SparseConvTensor) -> IndiceData:
        if self.indice_key is not None and self.indice_key in x.indice_dict:
            return x.indice_dict[self.indice_key]
        data = generate_subm_indice_pairs(
            x.indices,
            tuple(x.spatial_shape),
            x.batch_size,
            self.kernel_size,
            self.dilation,
        )
        if self.indice_key is not None:
            x.indice_dict[self.indice_key] = data
        return data

    def _get_or_compute_strided_indice(self, x: SparseConvTensor) -> IndiceData:
        if self.indice_key is not None and self.indice_key in x.indice_dict:
            return x.indice_dict[self.indice_key]
        data = generate_strided_indice_pairs(
            x.indices,
            tuple(x.spatial_shape),
            x.batch_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
        if self.indice_key is not None:
            x.indice_dict[self.indice_key] = data
        return data

    def _forward_subm(self, x: SparseConvTensor) -> SparseConvTensor:
        indice_data = self._get_or_compute_subm_indice(x)
        weight = self._get_weight()
        N_out = len(indice_data.out_indices)
        out_features = sparse_conv_forward(x.features, weight, self.bias, indice_data, N_out)
        return SparseConvTensor(
            out_features,
            indice_data.out_indices,
            x.spatial_shape,
            x.batch_size,
            indice_dict=x.indice_dict,
        )

    def _forward_strided(self, x: SparseConvTensor) -> SparseConvTensor:
        indice_data = self._get_or_compute_strided_indice(x)
        weight = self._get_weight()
        N_out = len(indice_data.out_indices)
        out_features = sparse_conv_forward(x.features, weight, self.bias, indice_data, N_out)
        return SparseConvTensor(
            out_features,
            indice_data.out_indices,
            list(indice_data.output_spatial_shape),
            x.batch_size,
            indice_dict=x.indice_dict,
        )

    def _forward_inverse(self, x: SparseConvTensor) -> SparseConvTensor:
        if self.indice_key is None:
            raise ValueError("SparseInverseConv3d requires indice_key")
        if self.indice_key not in x.indice_dict:
            raise KeyError(
                f"indice_key '{self.indice_key}' not found in indice_dict. "
                "The paired SparseConv3d must run before SparseInverseConv3d."
            )
        fwd: IndiceData = x.indice_dict[self.indice_key]

        weight = self._get_weight()

        # spconv k=1 special path: no coordinate expansion; output stays in the same
        # spatial space as the inverse input (mirrors spconv's own 1x1x1 fast path).
        kD, kH, kW = fwd.kernel_size
        if kD == 1 and kH == 1 and kW == 1:
            out_features = x.features @ weight[0]
            if self.bias is not None:
                out_features = out_features + self.bias
            return SparseConvTensor(
                out_features,
                x.indices,
                x.spatial_shape,
                x.batch_size,
                indice_dict=x.indice_dict,
            )

        # General case: swap pairs so forward input voxels become inverse output voxels.
        # Only emit output voxels that participated in ≥1 forward kernel pair
        # (matches spconv semantics: voxels with no forward connections are not restored).
        all_fwd_in = [p for p in fwd.pairs_in if p.numel() > 0]
        if all_fwd_in:
            participating = torch.cat(all_fwd_in).unique()
        else:
            participating = torch.empty(0, dtype=torch.long, device=fwd.input_indices.device)

        # Build compact id remapping: original_input_id -> compact_output_id
        compact_size = participating.numel()
        id_map = torch.full(
            (len(fwd.input_indices),), -1,
            dtype=torch.long, device=fwd.input_indices.device,
        )
        if compact_size > 0:
            id_map[participating] = torch.arange(compact_size, device=fwd.input_indices.device)

        inv_pairs_in = fwd.pairs_out
        inv_pairs_out = [id_map[p] if p.numel() > 0 else p for p in fwd.pairs_in]

        out_indices = fwd.input_indices[participating] if compact_size > 0 \
            else torch.empty((0, 4), dtype=torch.int32, device=fwd.input_indices.device)

        inv_data = IndiceData(
            out_indices=out_indices,
            pairs_in=inv_pairs_in,
            pairs_out=inv_pairs_out,
            pair_num=fwd.pair_num,
            input_indices=fwd.out_indices,
            input_spatial_shape=fwd.output_spatial_shape,
            output_spatial_shape=fwd.input_spatial_shape,
            kernel_size=fwd.kernel_size,
            stride=fwd.stride,
            padding=fwd.padding,
            dilation=fwd.dilation,
            subm=False,
        )

        out_features = sparse_conv_forward(x.features, weight, self.bias, inv_data, compact_size)
        return SparseConvTensor(
            out_features,
            out_indices,
            list(fwd.input_spatial_shape),
            x.batch_size,
            indice_dict=x.indice_dict,
        )


class SubMConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
        indice_key: Optional[str] = None,
        algo=None,
    ):
        super().__init__(
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
            subm=True,
            indice_key=indice_key,
            algo=algo,
        )


class SparseConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
        indice_key: Optional[str] = None,
        algo=None,
    ):
        super().__init__(
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            subm=False,
            indice_key=indice_key,
            algo=algo,
        )


class SparseInverseConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        bias: bool = True,
        indice_key: Optional[str] = None,
        algo=None,
    ):
        super().__init__(
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
            subm=False,
            inverse=True,
            indice_key=indice_key,
            algo=algo,
        )


class SparseSequential(SparseModule):
    """Sequential container that handles SparseConvTensor passing through dense layers."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for key, module in kwargs.items():
            self.add_module(key, module)

    def forward(self, x: Union[SparseConvTensor, torch.Tensor]) -> Union[SparseConvTensor, torch.Tensor]:
        for module in self.children():
            if isinstance(x, SparseConvTensor):
                if isinstance(module, SparseModule):
                    x = module(x)
                else:
                    x = x.replace_feature(module(x.features))
            else:
                x = module(x)
        return x
