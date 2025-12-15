# TGL Kernel

Custom CUDA kernels for Multi-head Latent Attention (MLA) in DeepSeek-V2/V3 models.

## Overview

TGL Kernel is an independent library containing two high-performance CUDA kernels originally adapted from TensorRT-LLM and FlashInfer:

1. **K Transform Kernel** - Efficient transformation for MLA attention
2. **MLA FP8 Quantization** - FP8 quantization for Q, K, V tensors

## Features

- ⚡ **High Performance**: Optimized CUDA kernels with vectorized memory access
- 🔧 **Flexible Compilation**: Supports both JIT and AOT compilation
- 📦 **Easy Installation**: `pip install` from git with automatic compilation
- 🎯 **Focused**: Only the kernels you need, minimal dependencies

## Installation

### From Git (Recommended)

```bash
pip install git+https://github.com/your-org/tgl-kernel.git
```

### From Source (Development)

```bash
git clone https://github.com/your-org/tgl-kernel.git
cd tgl-kernel
pip install -e .
```

### With AOT Compilation (for wheel distribution)

```bash
TGL_KERNEL_AOT_COMPILE=1 pip install .
```

## Quick Start

### K Transform

```python
import torch
from tgl_kernel import k_transform

# Input shapes:
# kv: [M, 8192] - interleaved k_nope and v data
# k_pe: [M, 64] - positional encoding

kv = torch.randn([1024, 8192], device="cuda", dtype=torch.bfloat16)
k_pe = torch.randn([1024, 64], device="cuda", dtype=torch.bfloat16)

# Transform to K: [M, 32, 192]
K = k_transform(kv, k_pe)
```

### MLA FP8 Quantization

```python
import torch
from tgl_kernel import mla_context_fp8_quantize

q = torch.randn([1024, 32, 192], device="cuda", dtype=torch.bfloat16)
k = torch.randn([1024, 32, 192], device="cuda", dtype=torch.bfloat16)

# V must have non-contiguous MLA layout (slice from kv_combined)
kv_combined = torch.randn([1024, 32, 256], device="cuda", dtype=torch.bfloat16)
v = kv_combined[:, :, 128:]  # Non-contiguous slice

# Quantize to FP8
q_fp8, k_fp8, v_fp8, bmm1_scale, bmm2_scale = mla_context_fp8_quantize(q, k, v)
```

## Requirements

- Python >= 3.10
- PyTorch with CUDA support
- CUDA >= 11.0
- GPU with compute capability >= 7.5 (sm75+)

## Architecture

```
tgl-kernel/
├── tgl_kernel/
│   ├── csrc/              # CUDA kernel implementations
│   ├── python/            # Python API wrappers
│   ├── jit/               # JIT compilation system
│   └── aot/               # AOT compilation support
├── tests/                 # Unit tests
├── benchmarks/            # Performance benchmarks
└── examples/              # Usage examples
```

## Compilation Modes

### JIT Compilation (Default)

Kernels are compiled on first use and cached:

```python
import tgl_kernel
# First call triggers JIT compilation
K = tgl_kernel.k_transform(kv, k_pe)
```

### AOT Compilation

Pre-compile kernels during installation:

```bash
# Enable AOT compilation
export TGL_KERNEL_AOT_COMPILE=1
pip install .
```

## Environment Variables

- `TGL_KERNEL_CUDA_ARCH_LIST`: Specify target CUDA architectures (e.g., "8.0 8.9 9.0")
- `TGL_KERNEL_AOT_COMPILE`: Enable AOT compilation (0 or 1)
- `TGL_KERNEL_DISABLE_JIT`: Disable JIT compilation, require AOT modules
- `TGL_KERNEL_LOGGING_LEVEL`: Set logging level (debug, info, warning, error)
- `TGL_KERNEL_WORKSPACE_BASE`: Override workspace directory

## Development

### Running Tests

```bash
# Install with editable mode
pip install -e .

# Run tests
python tests/test_k_transform.py
python tests/test_mla_fp8_quant.py
```

### Running Benchmarks

```bash
# Install required packages
pip install tabulate

# Run benchmarks (requires flashinfer/private/benchmarks)
python benchmarks/benchmark_mla_fp8.py
```

## Performance

Both kernels are optimized for modern NVIDIA GPUs:

- **K Transform**: Vectorized uint4 (16-byte) loads, shared memory for broadcasting
- **MLA FP8 Quantization**: Float4 vectorized access, per-tensor quantization

Typical performance on A100:
- K Transform: ~1-2ms for 1024 tokens
- FP8 Quantization: ~2-3ms for 1024 tokens with 32 heads

## Kernel Details

### K Transform Kernel

Transforms interleaved KV tensor and positional encoding into K tensor for MLA attention:

- Input: `kv [M, 8192]`, `k_pe [M, 64]`
- Output: `K [M, 32, 192]`
- Operations:
  1. Extract `k_nope` from first 4096 elements (32 heads × 128 dims)
  2. Broadcast `k_pe` to all 32 heads
  3. Concatenate per head: `k_nope [128] + k_pe [64] = K [192]`

### MLA FP8 Quantization

Quantizes Q, K, V tensors to FP8 (e4m3) format with scale computation:

- Input: Q, K, V in fp16/bf16/fp32
- Output: Q, K, V in fp8_e4m3 + BMM scales
- Features:
  - Per-tensor quantization
  - Handles non-contiguous V tensor
  - Computes BMM1/BMM2 scales for attention

## Credits

- **K Transform Kernel**: Adapted from TensorRT-LLM's MLA implementation
- **MLA FP8 Quantization**: Adapted from TensorRT-LLM's FP8 quantization kernels
- **JIT System**: Inspired by FlashInfer's compilation infrastructure

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use TGL Kernel in your research, please cite:

```bibtex
@software{tgl_kernel,
  title = {TGL Kernel: Custom CUDA Kernels for Multi-head Latent Attention},
  author = {TGL team},
  year = {2025},
  url = {https://github.com/your-org/tgl-kernel}
}
```

## Related Projects

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - Kernel library for LLM serving
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA's TensorRT for LLMs
- [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2) - Multi-head Latent Attention architecture
