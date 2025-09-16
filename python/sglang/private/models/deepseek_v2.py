import torch

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.fp8_kernel import (
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import AttnForwardMethod
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA as SGLANG_DeepseekV2AttentionMLA,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    use_intel_amx_backend,
)

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_gfx95_supported = is_gfx95_supported()
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
_is_cuda = is_cuda()

if _use_aiter_gfx95:
    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat

if _is_cuda:
    from sgl_kernel import bmm_fp8


class DeepseekV2AttentionMLA(SGLANG_DeepseekV2AttentionMLA):
    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        def _dispatch_mla_subtype():
            if _is_hip:
                if (
                    self.rocm_fused_decode_mla
                    and forward_batch.forward_mode.is_decode()
                ):
                    return AttnForwardMethod.MLA_FUSED_ROPE
                else:
                    return AttnForwardMethod.MLA
            else:
                if hasattr(self, "fused_qkv_a_proj_with_mqa") and use_intel_amx_backend(
                    self
                ):
                    return AttnForwardMethod.MLA_FUSED_ROPE_CPU
                else:
                    return AttnForwardMethod.MLA

        # Determine attention backend used by current forward batch
        if forward_batch.forward_mode.is_decode_or_idle():
            attention_backend = global_server_args_dict["decode_attention_backend"]
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            # Use the specified backend for speculative operations (both verify and draft extend)
            if global_server_args_dict["speculative_attention_mode"] == "decode":
                attention_backend = global_server_args_dict["decode_attention_backend"]
            else:  # default to prefill
                attention_backend = global_server_args_dict["prefill_attention_backend"]
        else:
            attention_backend = global_server_args_dict["prefill_attention_backend"]
        self.current_attention_backend = attention_backend

        if attention_backend == "ascend":
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                return AttnForwardMethod.MHA
            else:
                return AttnForwardMethod.MLA
        elif (
            attention_backend == "flashinfer"
            or attention_backend == "fa3"
            or attention_backend == "flashmla"
            or attention_backend == "cutlass_mla"
        ):
            # Use MHA with chunked KV cache when prefilling on long sequences.
            sum_extend_prefix_lens = (
                sum(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.extend_prefix_lens_cpu is not None
                else 0
            )
            # Flashinfer MLA: Do not absorb when enabling ragged prefill
            disable_ragged = (
                attention_backend == "flashinfer" or attention_backend == "flashmla"
            ) and self.flashinfer_mla_disable_ragged
            if (
                not disable_ragged
                and forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and (
                    (
                        sum_extend_prefix_lens >= self.chunked_prefix_cache_threshold
                        and not self.disable_chunked_prefix_cache
                    )
                    or sum_extend_prefix_lens == 0
                )
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return _dispatch_mla_subtype()
        elif attention_backend == "trtllm_mla":
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return _dispatch_mla_subtype()
        elif attention_backend == "trtllm_mla_tgl":
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return _dispatch_mla_subtype()
        elif attention_backend == "aiter":
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                if is_dp_attention_enabled():
                    if sum(forward_batch.extend_prefix_lens_cpu) == 0:
                        return AttnForwardMethod.MHA
                    else:
                        return AttnForwardMethod.MLA
                else:
                    return AttnForwardMethod.MHA
            else:
                return AttnForwardMethod.MLA
        else:
            # Triton: Use normal computation for prefill and use weight absorption for extend/decode
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return _dispatch_mla_subtype()

    def forward_absorb_core(
        self, q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator, positions
    ):
        if (
            self.current_attention_backend == "fa3"
            or self.current_attention_backend == "flashinfer"
            or self.current_attention_backend == "cutlass_mla"
            or self.current_attention_backend == "trtllm_mla"
            or self.current_attention_backend == "trtllm_mla_tgl"
            or self.current_attention_backend == "ascend"
        ):
            extra_args = {}
            if self._fuse_rope_for_trtllm_mla(forward_batch):
                extra_args = {
                    "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                    "is_neox": self.rotary_emb.is_neox_style,
                }
            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
                **extra_args,
            )
        else:
            if _use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache
                q, k = fused_qk_rope_cat(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    positions,
                    cos,
                    sin,
                    self.rotary_emb.is_neox_style,
                )
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            attn_output = self.attn_mqa(q, k, k_nope, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )

            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            attn_bmm_output = torch.empty(
                (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                dtype=attn_output.dtype,
                device=attn_output.device,
            )
            torch.bmm(
                attn_output.transpose(0, 1),
                self.w_vc,
                out=attn_bmm_output.view(
                    -1, self.num_local_heads, self.v_head_dim
                ).transpose(0, 1),
            )
        output, _ = self.o_proj(attn_bmm_output)

        return output

    def _fuse_rope_for_trtllm_mla(self, forward_batch: ForwardBatch) -> bool:
        """
        Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in fp8_e4m3 path.
        """
        if global_server_args_dict["decode_attention_backend"] == "trtllm_mla_tgl":
            return (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ) and forward_batch.attn_backend.data_type == torch.float8_e4m3fn
        else:
            return (
                self.current_attention_backend == "trtllm_mla"
                and forward_batch.forward_mode.is_decode_or_idle()
                and forward_batch.attn_backend.data_type == torch.float8_e4m3fn
            )
