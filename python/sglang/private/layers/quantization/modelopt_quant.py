from typing import Optional

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod as SGLANG_ModelOptFp4LinearMethod,
)
from sglang.srt.utils import is_cuda

if is_cuda():
    from sgl_kernel import scaled_fp4_quant

try:
    from flashinfer import mm_fp4 as fp4_gemm

    enable_flashinfer_fp4_gemm = True
except ImportError:
    if is_cuda():
        from sgl_kernel import cutlass_scaled_fp4_mm as fp4_gemm
    else:
        fp4_gemm = None
    enable_flashinfer_fp4_gemm = False


class ModelOptFp4LinearMethod(SGLANG_ModelOptFp4LinearMethod):

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        x_m, _ = x.shape
        w_n, _ = layer.weight.shape
        output_shape = [x_m, w_n]

        # Quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_scale_interleaved = scaled_fp4_quant(x, layer.input_scale_inv)

        assert x_fp4.dtype == torch.uint8
        assert x_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.weight.dtype == torch.uint8
        assert layer.weight_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        w = layer.weight
        w_scale_interleaved = layer.weight_scale_interleaved
        if enable_flashinfer_fp4_gemm:
            w = layer.weight.T
            w_scale_interleaved = layer.weight_scale_interleaved.T
        out = fp4_gemm(
            x_fp4,
            w,
            x_scale_interleaved,
            w_scale_interleaved,
            layer.alpha,
            output_dtype,
            backend="cutlass",
        )
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
