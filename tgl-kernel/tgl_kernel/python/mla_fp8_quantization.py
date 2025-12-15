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

from typing import Optional, Tuple

import torch

from ..jit import get_mla_fp8_quantization_module


def mla_context_fp8_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    quant_scale_qkv: Optional[torch.Tensor] = None,
    quant_scale_o: Optional[torch.Tensor] = None,
    dequant_scale_q: Optional[torch.Tensor] = None,
    dequant_scale_kv: Optional[torch.Tensor] = None,
    host_bmm1_scale: float = 1.0,
    compute_scales: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Quantize Q, K, V tensors from high precision (float32/float16/bfloat16) to FP8 (e4m3)
    for Multi-head Latent Attention (MLA) context phase.

    This function is designed for MLA architecture where Q and K have dimensions
    [batch_tokens, num_heads, qk_nope_dim + qk_rope_dim] and V has dimension
    [batch_tokens, num_heads, v_dim].

    Args:
        q: Query tensor of shape [total_q_len, num_heads, 192]
           where 192 = qk_nope_dim (128) + qk_rope_dim (64)
        k: Key tensor of shape [total_kv_len, num_heads, 192]
        v: Value tensor of shape [total_kv_len, num_heads, 128]
           IMPORTANT: V MUST be non-contiguous from kv[:, :, 128:] slice
           with stride (256*num_heads, 256, 1). Contiguous V is not supported.
        quant_scale_qkv: Quantization scale for Q/K/V, shape [1], default 1.0
        quant_scale_o: Quantization scale for output, shape [1], default 1.0
        dequant_scale_q: Dequantization scale for Q, shape [1], default 1.0
        dequant_scale_kv: Dequantization scale for K/V, shape [1], default 1.0
        host_bmm1_scale: Host-side BMM1 scale factor (e.g., 1/sqrt(head_dim))

    Returns:
        Tuple containing:
        - quant_q: Quantized Q tensor in FP8, shape [total_q_len, num_heads, 192]
        - quant_k: Quantized K tensor in FP8, shape [total_kv_len, num_heads, 192]
        - quant_v: Quantized V tensor in FP8, shape [total_kv_len, num_heads, 128]
        - bmm1_scale: BMM1 scale factors, shape [2]
        - bmm2_scale: BMM2 scale factor, shape [1]

    Example:
        >>> import torch
        >>> from flashinfer.jit.mla_fp8_quantization import mla_context_fp8_quantize
        >>>
        >>> # MLA dimensions for DeepSeek-V3
        >>> total_q_len = 1024
        >>> total_kv_len = 1024
        >>> num_heads = 64
        >>> qk_head_dim = 192  # 128 nope + 64 rope
        >>> v_head_dim = 128
        >>>
        >>> # Create input tensors
        >>> q = torch.randn(total_q_len, num_heads, qk_head_dim,
        ...                 dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(total_kv_len, num_heads, qk_head_dim,
        ...                 dtype=torch.bfloat16, device='cuda')
        >>> # V must be non-contiguous from MLA layout
        >>> kv_combined = torch.randn(total_kv_len, num_heads, 256,
        ...                            dtype=torch.bfloat16, device='cuda')
        >>> v = kv_combined[:, :, 128:]  # Non-contiguous slice
        >>>
        >>> # Quantize
        >>> quant_q, quant_k, quant_v, bmm1_scale, bmm2_scale = mla_context_fp8_quantize(
        ...     q, k, v,
        ...     host_bmm1_scale=1.0 / (qk_head_dim ** 0.5)
        ... )
    """
    # Get the JIT compiled module
    _module = get_mla_fp8_quantization_module()

    # Validate input tensors
    assert q.is_cuda, "q must be on CUDA device"
    assert k.is_cuda, "k must be on CUDA device"
    assert v.is_cuda, "v must be on CUDA device"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    assert q.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported dtype: {q.dtype}, must be float32, float16, or bfloat16"

    # Ensure Q and K are contiguous
    # V MUST be non-contiguous from MLA layout (kernel requires this)
    q = q.contiguous()
    k = k.contiguous()
    # v must NOT be contiguous - validate stride below

    # Validate dimensions
    total_q_len, num_heads, qk_head_dim = q.shape
    total_kv_len = k.shape[0]
    v_head_dim = v.shape[2]

    assert (
        qk_head_dim == 192
    ), f"qk_head_dim must be 192 (128 nope + 64 rope), got {qk_head_dim}"
    assert v_head_dim == 128, f"v_head_dim must be 128, got {v_head_dim}"
    assert k.shape == (
        total_kv_len,
        num_heads,
        qk_head_dim,
    ), f"k shape mismatch: expected {(total_kv_len, num_heads, qk_head_dim)}, got {k.shape}"
    assert v.shape == (
        total_kv_len,
        num_heads,
        v_head_dim,
    ), f"v shape mismatch: expected {(total_kv_len, num_heads, v_head_dim)}, got {v.shape}"

    # Validate V stride for non-contiguous MLA layout
    # V MUST come from slicing a [batch, heads, 256] tensor at [:, :, 128:]
    # This results in stride (256*heads, 256, 1)
    # Contiguous V is NOT supported by this kernel
    v_stride = v.stride()
    QK_NOPE_HEAD_DIM = 128  # Must match the CUDA kernel constant

    # Check if V has the expected non-contiguous stride pattern
    # Expected: stride[1] = QK_NOPE_HEAD_DIM + V_HEAD_DIM = 256
    expected_head_stride = QK_NOPE_HEAD_DIM + v_head_dim  # 128 + 128 = 256
    expected_token_stride = expected_head_stride * num_heads  # 256 * num_heads

    # V must be non-contiguous from MLA layout: stride = (256*heads, 256, 1)
    is_mla_noncontiguous = (
        v_stride[1] == expected_head_stride
        and v_stride[0] == expected_token_stride
        and v_stride[2] == 1
    )

    if not is_mla_noncontiguous:
        raise ValueError(
            f"V tensor must have non-contiguous MLA layout.\n"
            f"Expected stride: ({expected_token_stride}, {expected_head_stride}, 1)\n"
            f"Got stride: {v_stride}, shape: {v.shape}\n"
            f"To create V correctly:\n"
            f"  kv_combined = torch.randn({total_kv_len}, {num_heads}, 256, dtype=dtype, device='cuda')\n"
            f"  v = kv_combined[:, :, 128:]  # Non-contiguous slice"
        )

    # Allocate output tensors
    # Use torch.float8_e4m3fn for proper FP8 type
    quant_q = torch.empty(
        (total_q_len, num_heads, qk_head_dim),
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    quant_k = torch.empty(
        (total_kv_len, num_heads, qk_head_dim),
        dtype=torch.float8_e4m3fn,
        device=k.device,
    )
    quant_v = torch.empty(
        (total_kv_len, num_heads, v_head_dim),
        dtype=torch.float8_e4m3fn,
        device=v.device,
    )

    # Allocate scale tensors only if needed
    if compute_scales:
        bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
        bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    else:
        # Pass empty tensors (will be nullptr in C++)
        bmm1_scale = torch.empty(0, dtype=torch.float32, device=q.device)
        bmm2_scale = torch.empty(0, dtype=torch.float32, device=q.device)

    # Prepare scale tensors (convert None to empty tensor for FFI)
    if quant_scale_qkv is None:
        quant_scale_qkv = torch.empty(0, dtype=torch.float32, device=q.device)
    if quant_scale_o is None:
        quant_scale_o = torch.empty(0, dtype=torch.float32, device=q.device)
    if dequant_scale_q is None:
        dequant_scale_q = torch.empty(0, dtype=torch.float32, device=q.device)
    if dequant_scale_kv is None:
        dequant_scale_kv = torch.empty(0, dtype=torch.float32, device=q.device)

    # Ensure scales are on the same device
    if quant_scale_qkv.numel() > 0:
        quant_scale_qkv = quant_scale_qkv.to(q.device)
    if quant_scale_o.numel() > 0:
        quant_scale_o = quant_scale_o.to(q.device)
    if dequant_scale_q.numel() > 0:
        dequant_scale_q = dequant_scale_q.to(q.device)
    if dequant_scale_kv.numel() > 0:
        dequant_scale_kv = dequant_scale_kv.to(q.device)

    # Call the kernel
    _module.mla_context_fp8_quantize(
        q,
        quant_q,
        k,
        quant_k,
        v,
        quant_v,
        bmm1_scale,
        bmm2_scale,
        quant_scale_qkv,
        quant_scale_o,
        dequant_scale_q,
        dequant_scale_kv,
        float(host_bmm1_scale),
    )

    # Return None for scales if not computed
    if compute_scales:
        return quant_q, quant_k, quant_v, bmm1_scale, bmm2_scale
    else:
        return quant_q, quant_k, quant_v, None, None


__all__ = [
    "mla_context_fp8_quantize",
]
