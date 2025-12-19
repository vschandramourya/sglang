"""
Copyright (c) 2025 by TGL team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools

import torch

from ..jit import get_k_transform_module


@functools.cache
def _get_k_transform_module():
    """Get or create the k_transform module."""
    return get_k_transform_module()


def k_transform(
    v_full: torch.Tensor,
    k_pe: torch.Tensor,
    num_heads,
) -> torch.Tensor:
    r"""Transform kernel: Extract k_nope from v_full and broadcast k_pe to form K.

    This kernel performs the following transformation for MLA (Multi-head Latent Attention):
    1. Extracts k_nope from the first 4096 elements of v_full [M, 8192]
    2. Broadcasts k_pe [M, 64] to all 32 heads
    3. Concatenates k_nope + k_pe per head to form K [M, 32, 192]

    Parameters
    ----------
    v_full: torch.Tensor
        Input tensor [M, 8192] containing k_nope in first 4096 elements (32 heads * 128 dims).
        Supported dtypes: float16, bfloat16, float32.

    k_pe: torch.Tensor
        Positional encoding tensor [M, 64] to broadcast to all heads.
        Must have the same dtype and device as v_full.

    Returns
    -------
    K: torch.Tensor
        Output tensor [M, 32, 192] = [M, NUM_HEADS, QK_NOPE_DIM + QK_ROPE_DIM]
        where each head has k_nope (128 dims) + k_pe (64 dims).

    Notes
    -----
    - This kernel is optimized for MLA attention with fixed dimensions:
      * NUM_HEADS = 32
      * QK_NOPE_DIM = 128 (non-positional key dimension)
      * QK_ROPE_DIM = 64 (positional encoding dimension)
      * V_HEAD_DIM = 128
      * V_FULL_DIM = 8192 = 32 * (128 + 128)
    - The kernel uses vectorized loads (uint4) and shared memory for efficiency.
    - Each warp processes one row, with 4 warps per block.

    Examples
    --------
    >>> import torch
    >>> from flashinfer import k_transform
    >>> M = 1024  # sequence length
    >>> v_full = torch.randn([M, 8192], device="cuda", dtype=torch.bfloat16)
    >>> k_pe = torch.randn([M, 64], device="cuda", dtype=torch.bfloat16)
    >>> K = k_transform(v_full, k_pe)
    >>> K.shape
    torch.Size([1024, 32, 192])
    """

    # Input validation
    if v_full.ndim != 2:
        raise ValueError(f"v_full must be 2D tensor, got shape {v_full.shape}")
    if k_pe.ndim != 2:
        raise ValueError(f"k_pe must be 2D tensor, got shape {k_pe.shape}")

    M = v_full.shape[0]
    if k_pe.shape[0] != M:
        raise ValueError(
            f"v_full and k_pe must have same batch dimension, "
            f"got v_full.shape[0]={M}, k_pe.shape[0]={k_pe.shape[0]}"
        )

    # Check dimensions
    QK_NOPE_DIM = 128
    QK_ROPE_DIM = 64
    K_DIM_PER_HEAD = QK_NOPE_DIM + QK_ROPE_DIM
    V_HEAD_DIM = 128

    # Check dtype
    if v_full.dtype != k_pe.dtype:
        raise ValueError(
            f"v_full and k_pe must have same dtype, "
            f"got v_full.dtype={v_full.dtype}, k_pe.dtype={k_pe.dtype}"
        )

    # Check device
    if v_full.device != k_pe.device:
        raise ValueError(
            f"v_full and k_pe must be on same device, "
            f"got v_full.device={v_full.device}, k_pe.device={k_pe.device}"
        )

    # Check supported dtypes
    if v_full.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        raise ValueError(
            f"Unsupported dtype {v_full.dtype}. "
            f"Supported dtypes: float16, bfloat16, float32"
        )

    # Allocate output tensor
    K = torch.empty(
        M, num_heads, K_DIM_PER_HEAD, dtype=v_full.dtype, device=v_full.device
    )

    # Get module and run kernel
    mod = _get_k_transform_module()
    mod.k_transform(K, v_full, k_pe)

    return K
