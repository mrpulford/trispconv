"""Generate golden reference outputs using the system-installed spconv.

Run once (requires spconv installed):
    python tests/generate_spconv_goldens.py

Writes fixtures to tests/fixtures/spconv_golden_*.pt
Each fixture is a dict with keys:
  features, indices, spatial_shape, batch_size  -- input
  weight, bias                                   -- module params (KRSC layout)
  kernel_size, stride, padding, dilation         -- conv params
  out_features, out_indices, out_spatial_shape   -- spconv output (sorted by coord)
"""
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import spconv.pytorch as spconv

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(exist_ok=True)


def _encode_coords(indices, spatial_shape):
    D, H, W = spatial_shape
    idx = indices.long()
    return idx[:, 0] * (D * H * W) + idx[:, 1] * (H * W) + idx[:, 2] * W + idx[:, 3]


def _sort_by_coord(features, indices, spatial_shape):
    """Return features and indices sorted by encoded coordinate (deterministic order)."""
    enc = _encode_coords(indices, spatial_shape)
    order = enc.argsort()
    return features[order], indices[order]


def _make_input(shape, batch_size, n_per_batch, C_in, seed):
    rng = np.random.default_rng(seed)
    D, H, W = shape
    all_coords = np.stack(
        np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    batch_indices = []
    for b in range(batch_size):
        perm = rng.permutation(len(all_coords))[:n_per_batch]
        inds = all_coords[perm]
        batch_col = np.full((len(inds), 1), b)
        batch_indices.append(np.concatenate([batch_col, inds], axis=1))
    indices = torch.from_numpy(np.concatenate(batch_indices).astype(np.int32))
    features = torch.from_numpy(rng.uniform(-1, 1, (len(indices), C_in)).astype(np.float32))
    return features, indices


def _run_spconv(module, features, indices, spatial_shape, batch_size, indice_dict=None):
    x = spconv.SparseConvTensor(features, indices, list(spatial_shape), batch_size,
                                indice_dict=indice_dict or {})
    with torch.no_grad():
        y = module(x)
    return y.features, y.indices, tuple(y.spatial_shape)


CASES = [
    # (tag, conv_type, C_in, C_out, k, stride, padding, dilation, bias)
    ("subm_3x3x3_c16",   "subm",    16, 16, 3, 1, 0, 1, True),
    ("subm_3x3x3_c32",   "subm",    32, 32, 3, 1, 0, 1, True),
    ("subm_1x1x1_c16",   "subm",    16, 32, 1, 1, 0, 1, False),
    ("subm_3x3x3_d2",    "subm",    16, 16, 3, 1, 0, 2, True),
    ("strided_s2_p1",    "strided", 16, 32, 3, 2, 1, 1, True),
    ("strided_s2_p0",    "strided", 16, 32, 2, 2, 0, 1, False),
    ("strided_s1_k2",    "strided", 16, 16, 2, 1, 0, 1, True),
    ("inverse_3x3x3",    "inverse", 32, 16, 3, 2, 1, 1, True),
    ("inverse_1x1x1",    "inverse", 32, 16, 1, 2, 0, 1, False),
]

SHAPE = (12, 12, 12)
BATCH_SIZE = 2
N_PER_BATCH = 80


def generate_all():
    torch.manual_seed(0)

    for case in CASES:
        tag, conv_type, C_in, C_out, k, s, p, d, use_bias = case
        rng = np.random.default_rng(abs(hash(tag)) % (2**31))
        seed = abs(hash(tag)) % (2**31)

        features, indices = _make_input(SHAPE, BATCH_SIZE, N_PER_BATCH, C_in, seed)

        # KRSC weight: (C_out, k, k, k, C_in)
        w_np = rng.uniform(-1, 1, (C_out, k, k, k, C_in)).astype(np.float32)
        weight = torch.from_numpy(w_np)
        bias_val = torch.from_numpy(rng.uniform(-0.1, 0.1, C_out).astype(np.float32)) if use_bias else None

        algo = spconv.ConvAlgo.Native

        if conv_type == "subm":
            module = spconv.SubMConv3d(C_in, C_out, k, dilation=d, bias=use_bias, algo=algo)
            module.weight.data.copy_(weight)
            if use_bias:
                module.bias.data.copy_(bias_val)
            with torch.no_grad():
                out_f, out_i, out_shape = _run_spconv(module, features, indices, SHAPE, BATCH_SIZE)
            out_f_s, out_i_s = _sort_by_coord(out_f, out_i, out_shape)

        elif conv_type == "strided":
            module = spconv.SparseConv3d(C_in, C_out, k, stride=s, padding=p, dilation=d, bias=use_bias, algo=algo)
            module.weight.data.copy_(weight)
            if use_bias:
                module.bias.data.copy_(bias_val)
            with torch.no_grad():
                out_f, out_i, out_shape = _run_spconv(module, features, indices, SHAPE, BATCH_SIZE)
            out_f_s, out_i_s = _sort_by_coord(out_f, out_i, out_shape)

        elif conv_type == "inverse":
            # Need to first run a SparseConv3d to produce indice data, then inverse
            down = spconv.SparseConv3d(C_out, C_in, k, stride=s, padding=p, bias=False, indice_key="down", algo=algo)
            w_down_np = rng.uniform(-1, 1, (C_in, k, k, k, C_out)).astype(np.float32)
            down.weight.data.copy_(torch.from_numpy(w_down_np))

            # Run down to get intermediate + indice cache
            x_in = spconv.SparseConvTensor(
                torch.from_numpy(rng.uniform(-1, 1, (len(indices), C_out)).astype(np.float32)),
                indices, list(SHAPE), BATCH_SIZE
            )
            with torch.no_grad():
                x_down = down(x_in)

            # Now run the inverse on x_down features, but replace with our C_in features
            x_for_inv = spconv.SparseConvTensor(
                features[:len(x_down.indices)],   # match N_out of downsample
                x_down.indices,
                x_down.spatial_shape,
                BATCH_SIZE,
                indice_dict=x_down.indice_dict,
            )
            # Trim features to actual number of output voxels from down
            features_inv = torch.from_numpy(
                rng.uniform(-1, 1, (len(x_down.indices), C_in)).astype(np.float32)
            )
            x_for_inv = spconv.SparseConvTensor(
                features_inv, x_down.indices, x_down.spatial_shape,
                BATCH_SIZE, indice_dict=x_down.indice_dict,
            )
            module = spconv.SparseInverseConv3d(C_in, C_out, k, bias=use_bias, indice_key="down", algo=algo)
            module.weight.data.copy_(weight)
            if use_bias:
                module.bias.data.copy_(bias_val)

            with torch.no_grad():
                out_f, out_i, out_shape = _run_spconv(module, features_inv, x_down.indices, x_down.spatial_shape, BATCH_SIZE, indice_dict=x_down.indice_dict)
            # For inverse, re-collect actual input used
            features = features_inv
            indices = x_down.indices
            out_f_s, out_i_s = _sort_by_coord(out_f, out_i, out_shape)

            # Save the down indice dict so trispconv can recreate it
            # We store the paired forward conv params so the test can reconstruct
            fixture = {
                "conv_type": conv_type,
                "C_in": C_in, "C_out": C_out,
                "kernel_size": k, "stride": s, "padding": p, "dilation": d,
                "features": features,
                "indices": indices,
                "spatial_shape": list(x_down.spatial_shape),  # inverse input space
                "batch_size": BATCH_SIZE,
                "weight": weight,
                "bias": bias_val,
                "out_features": out_f_s,
                "out_indices": out_i_s,
                "out_spatial_shape": list(SHAPE),   # inverse restores original shape
                # For trispconv to recreate the indice_dict we need the forward down params
                "down_in_features": x_in.features,
                "down_in_indices": x_in.indices,
                "down_in_spatial_shape": list(SHAPE),  # forward down's INPUT space
                "down_weight": torch.from_numpy(w_down_np),
                "down_C_in": C_out,   # down layer: C_out->C_in of the inverse
                "down_C_out": C_in,
            }
            path = FIXTURE_DIR / f"spconv_golden_{tag}.pt"
            torch.save(fixture, path)
            print(f"  saved {path.name}  out_shape={out_shape}  N_out={len(out_f_s)}")
            continue

        fixture = {
            "conv_type": conv_type,
            "C_in": C_in, "C_out": C_out,
            "kernel_size": k, "stride": s, "padding": p, "dilation": d,
            "features": features,
            "indices": indices,
            "spatial_shape": list(SHAPE),
            "batch_size": BATCH_SIZE,
            "weight": weight,
            "bias": bias_val,
            "out_features": out_f_s,
            "out_indices": out_i_s,
            "out_spatial_shape": list(out_shape),
        }
        path = FIXTURE_DIR / f"spconv_golden_{tag}.pt"
        torch.save(fixture, path)
        print(f"  saved {path.name}  out_shape={out_shape}  N_out={len(out_f_s)}")


if __name__ == "__main__":
    print("Generating spconv golden fixtures...")
    generate_all()
    print("Done.")
