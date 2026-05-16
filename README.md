# trispconv

Drop-in PyTorch/Triton reimplementation of the `spconv.pytorch` API for sparse 3D convolution inference.

```python
# Before
import spconv.pytorch as spconv

# After
import trispconv.pytorch as spconv
```

One import line change. Everything else stays the same.

## What it is

`trispconv` re-implements the `spconv.pytorch` public surface (modules, tensor, weight layout) in pure Python + PyTorch, with optional Triton acceleration on GPU. The goal is a pinnable, inspectable alternative for projects that depend on spconv for inference — no CUDA extension compilation required.

**CPU is a first-class target.** The full forward pass works without a GPU.

## Installation

```bash
# CPU only (pure PyTorch)
pip install trispconv

# With Triton acceleration (Linux, NVIDIA or AMD)
pip install "trispconv[triton]"
```

For AMD (ROCm), install the ROCm build of Triton instead of the PyPI wheel:

```bash
pip install triton-rocm   # or the ROCm-tagged wheel matching your ROCm version
pip install trispconv
```

## Supported modules

| Module | Notes |
|---|---|
| `SubMConv3d` | SubManifold sparse convolution |
| `SparseConv3d` | Strided sparse convolution |
| `SparseInverseConv3d` | Transposed / inverse sparse convolution |
| `SparseSequential` | Sequential container; dense modules (BN, ReLU, …) receive `features` directly |
| `SparseConvTensor` | Sparse tensor wrapper |
| `ConvAlgo` | Enum stub (`Native`, `MaskImplicitGemm`, `MaskSplitImplicitGemm`) |

## Weight layout

Weights are stored in spconv's **KRSC layout**: `(C_out, kD, kH, kW, C_in)`.  
This matches checkpoints saved with `ALL_WEIGHT_IS_KRSC=True` (the spconv 2.x default) and is the layout used by the public `module.weight` parameter.

## GPU acceleration (Triton)

When Triton is available and the input tensor is on a CUDA device, `trispconv` replaces the per-kernel-slot PyTorch gather→GEMM→scatter loop with a fused Triton kernel. This matches the accumulation order of spconv's `ConvAlgo.Native`, so numerical results are equivalent.

The dispatch condition is:

```python
use_triton = TRITON_AVAILABLE and features.device.type == "cuda"
```

`device.type == "cuda"` is used rather than `tensor.is_cuda` because it is explicit about what is being tested. On AMD/ROCm, PyTorch reports the device type as `"cuda"` (the ROCm build uses HIP's CUDA-compatibility layer), so the Triton path activates on AMD cards without any special casing, provided the ROCm build of Triton is installed.

Block sizes are autotuned per `(N_k, C_in, C_out)` shape on first use.

## spconv compatibility notes

trispconv deliberately preserves two spconv quirks so existing checkpoints and model code work without modification:

**k=1×1×1 weight convention** — spconv folds the weight as `features @ weight.view(C_in, C_out)` rather than `features @ weight.T`. This produces a different result from `nn.Conv3d` with a 1×1×1 kernel. trispconv matches this behaviour.

**`SparseInverseConv3d` with k=1×1×1** — spconv's inverse conv fast path for 1×1×1 kernels returns the same coordinate set as the *inverse input* (the downsampled space), not the original spatial shape. trispconv matches this behaviour.

Both of these will be documented as known deviations once a version is released that users can pin against.

## Running tests

```bash
# Pure-PyTorch tests (no GPU, no spconv required)
pytest tests/test_indices_subm.py tests/test_vs_dense.py

# All tests including spconv golden fixtures (requires spconv installed)
python tests/generate_spconv_goldens.py   # run once
pytest tests/
```

## Roadmap

- [x] Phase 1 — pure-PyTorch correctness baseline (validated against dense conv3d + spconv goldens)
- [x] Phase 2 — per-kernel-slot Triton gather-GEMM-scatter (NVIDIA + AMD)
- [ ] Phase 3 — autotuned block sizes, benchmark harness
- [ ] Phase 4 — gradient support (training path)
