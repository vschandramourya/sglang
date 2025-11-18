import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging.version import Version

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()

if _is_cuda:
    if _is_flashinfer_available:
        from flashinfer.norm import fused_add_rmsnorm
    else:
        from sgl_kernel import fused_add_rmsnorm
    from sgl_kernel import rmsnorm

if _use_aiter:
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
elif _is_hip:
    import vllm
    from vllm._custom_ops import fused_add_rms_norm, rms_norm

    _vllm_version = Version(vllm.__version__)

logger = logging.getLogger(__name__)

if _is_npu:
    import torch_npu


class CursorRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        if _use_aiter:
            self._forward_method = self.forward_aiter
        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_INFERENCE"):
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[Tuple[torch.Tensor], None]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return (x, residual), None
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[Tuple[torch.Tensor], None]]:
        if residual is not None:
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            return (out, residual_out), None
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[Tuple[torch.Tensor], None]]:
        if residual is not None:
            residual_out = torch.empty_like(x)
            output = torch.empty_like(x)
            fused_add_rms_norm(
                output,
                x,
                residual,
                residual_out,
                self.weight.data,
                self.variance_epsilon,
            )
            return (output, residual_out), None
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[Tuple[torch.Tensor], None]]:
        if not x.is_contiguous():
            # NOTE: Remove this if aiter kernel supports discontinuous input
            x = x.contiguous()
        if residual is not None:
            if _vllm_version < Version("0.9"):
                fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
                return (x, residual), None
            else:
                residual_out = torch.empty_like(x)
                output = torch.empty_like(x)
                fused_add_rms_norm(
                    output,
                    x,
                    residual_out,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return (output, residual_out), None
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[Tuple[torch.Tensor], None]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        if residual is None:
            return x
        else:
            return (x, residual), None

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[Tuple[torch.Tensor], None]]:
        if _is_cpu_amx_available:
            if residual is not None:
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return (x, residual), None
            return torch.ops.sgl_kernel.rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual)
