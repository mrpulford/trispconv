"""Triton kernels for sparse convolution.

GPU path: fused gather-GEMM-scatter per kernel slot (matches ConvAlgo.Native order).
CPU / no-Triton: callers fall back to the pure-PyTorch loop in functional.py.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}),
        ],
        key=["N_k", "C_in", "C_out"],
        # Without restore_value, autotune would benchmark each config by calling
        # the kernel (which does atomic_add) multiple times on the same out_ptr,
        # causing massive accumulation of spurious values.
        restore_value=["out_ptr"],
    )
    @triton.jit
    def _gather_gemm_scatter_kernel(
        feat_ptr,       # float32 (N_in, C_in)
        w_ptr,          # float32 (C_in, C_out)  — one kernel slot
        out_ptr,        # float32 (N_out, C_out) — accumulation target
        in_ids_ptr,     # int64   (N_k,)
        out_ids_ptr,    # int64   (N_k,)
        N_k,            # runtime: number of gather/scatter pairs
        C_in,           # runtime: input channels
        C_out,          # runtime: output channels
        feat_stride,    # runtime: features.stride(0)  (== C_in when contiguous)
        out_stride,     # runtime: out.stride(0)        (== C_out when contiguous)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # ---- row tile (gather dimension) ----
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = m_offs < N_k

        # ---- column tile (C_out) ----
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = n_offs < C_out

        # Load gather indices; out-of-range positions are zeroed out so they
        # contribute 0 to the dot-product and are masked on the atomic-add.
        in_ids = tl.load(in_ids_ptr + m_offs, mask=mask_m, other=0)

        # Accumulate across the inner (C_in) dimension.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(C_in, BLOCK_K)):
            k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = k_offs < C_in

            # A: gathered rows from features  (BLOCK_M, BLOCK_K)
            A = tl.load(
                feat_ptr + in_ids[:, None] * feat_stride + k_offs[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            # B: weight columns               (BLOCK_K, BLOCK_N)
            B = tl.load(
                w_ptr + k_offs[:, None] * C_out + n_offs[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.dot(A, B, out_dtype=tl.float32)

        # Scatter-add into the output buffer.
        # out_ids within a single kernel slot are unique, so no intra-block
        # conflicts exist; atomic_add is still used for correctness across
        # concurrent blocks (different pid_n for the same pid_m tile).
        out_ids = tl.load(out_ids_ptr + m_offs, mask=mask_m, other=0)
        tl.atomic_add(
            out_ptr + out_ids[:, None] * out_stride + n_offs[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def gather_gemm_scatter(
    features: "torch.Tensor",
    weight_k: "torch.Tensor",
    out: "torch.Tensor",
    in_ids: "torch.Tensor",
    out_ids: "torch.Tensor",
) -> None:
    """In-place: out[out_ids] += features[in_ids] @ weight_k.

    weight_k : (C_in, C_out) — one kernel slot, already sliced by the caller.
    Operates on CUDA tensors; caller must ensure device placement.
    """
    import torch  # local to avoid circular import at module load

    N_k = in_ids.shape[0]
    if N_k == 0:
        return

    C_in = features.shape[1]
    C_out = weight_k.shape[1]

    # Triton pointer arithmetic requires contiguous tensors.
    features = features.contiguous()
    weight_k = weight_k.contiguous()
    # out is already contiguous (created with torch.zeros).

    # Indices must be int64 for Triton pointer arithmetic.
    in_ids  = in_ids.contiguous().to(torch.int64)
    out_ids = out_ids.contiguous().to(torch.int64)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(N_k, meta["BLOCK_M"]),
        triton.cdiv(C_out, meta["BLOCK_N"]),
    )

    _gather_gemm_scatter_kernel[grid](
        features, weight_k, out,
        in_ids, out_ids,
        N_k, C_in, C_out,
        features.stride(0), out.stride(0),
    )
