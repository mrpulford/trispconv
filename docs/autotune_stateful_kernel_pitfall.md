# Pitfall: Triton Autotune with Stateful (Accumulating) Kernels

## The Bug Class

`@triton.autotune` benchmarks every config candidate on the **first call** with
a new key. It does this by calling the kernel **multiple times** with the exact
same tensor arguments that the caller passed in.

If the kernel **writes to an output argument** (atomic_add, store, scatter) rather
than returning a fresh tensor, each benchmark trial accumulates into that tensor.
By the time the actual computation call runs, the output has been incremented
N_trials × N_configs times instead of once. Results are silently and massively wrong.

On subsequent calls with the same autotune key the benchmark does not re-run, so
those calls produce correct results. This makes the bug intermittent and very hard
to spot: unit tests that call the same shape twice in one session may pass; first-
call-per-shape failures (e.g. the first layer of a model) will always break.

## How to Detect It

Look for any `@triton.autotune`-decorated kernel that:
- takes a pointer argument it writes to (`atomic_add`, `tl.store`, scatter-add)
- that pointer is part of the accumulation target (not a scratch buffer)

The autotune `key` doesn't matter — any first call with a *new* key value will
trigger the benchmark and corrupt the output.

## The Fix

Add `restore_value=["<arg_name>"]` to the `@triton.autotune` call for every
output-pointer argument the kernel modifies in-place:

```python
@triton.autotune(
    configs=[...],
    key=["N_k", "C_in", "C_out"],
    restore_value=["out_ptr"],   # <-- saves/restores out_ptr between benchmark trials
)
@triton.jit
def my_kernel(out_ptr, ...):
    ...
    tl.atomic_add(out_ptr + ..., val)
```

`restore_value` saves the tensor contents before each trial and restores them
afterward, so every trial starts from the same baseline and the final real call
sees `out_ptr` in its correct pre-call state.

## Checklist for Auditing This Codebase

For every `@triton.autotune`-decorated kernel, ask:

1. Does the kernel write to any pointer argument? (search: `tl.atomic_add`,
   `tl.store`, `tl.atomic_xchg`, `tl.atomic_cas`)
2. Is that pointer argument an accumulation target shared with the caller?
3. If yes to both: is `restore_value` listing that argument name?

If 1+2 are true and 3 is false → **bug**.

## Instance Fixed in This Codebase

`trispconv/pytorch/triton_kernels.py` — `_gather_gemm_scatter_kernel`

- Writes via `tl.atomic_add(out_ptr + ..., acc, mask=...)`
- `out_ptr` is the caller's pre-allocated accumulation buffer (passed through
  `gather_gemm_scatter` → `sparse_conv_forward`)
- Fixed by adding `restore_value=["out_ptr"]` to the autotune decorator

Secondary fix in the same kernel: `tl.dot(A, B, out_dtype=tl.float32)` to prevent
fp16 accumulator overflow for large channel-reduction ops (C_in × kv > 3456,
matching spconv's `use_f32_as_accum` heuristic).
