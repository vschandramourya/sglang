from __future__ import annotations

"""
Support private version of trtllm_mla backend. Supports FP8 prefill attention.
"""

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend as SGLANG_TRTLLMMLABackend,
)
from sglang.srt.layers.attention.trtllm_mla_backend import _concat_mla_absorb_q_general
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_HAS_FP8_CONVERT_KERNEL = False
if not get_global_server_args().disable_mla_context_fp8_quantize_kernel:
    try:
        from tgl_kernel import mla_context_fp8_quantize

        _HAS_FP8_CONVERT_KERNEL = True
    except ImportError:
        logger.warning("Failed to import mla_context_fp8_quantize from tgl_kernel")
else:
    logger.info("mla_context_fp8_quantize kernel disabled by server args")


class TRTLLMMLABackend(SGLANG_TRTLLMMLABackend):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This will be set to True in the private model runner if --enable-trtllm-mla-fp8-prefill is set.
        self.enable_fp8_prefill = get_global_server_args().enable_trtllm_mla_fp8_prefill
        if self.enable_fp8_prefill:
            logger.info("Enabled FP8 prefill attention for TRTLLM MLA backend")

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if (
            not self.enable_fp8_prefill
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

        # FP8 prefill PATH
        merge_query = q_rope is not None

        # Save KV cache if requested
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        # TODO refactor to avoid code duplication
        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q = _concat_mla_absorb_q_general(q_nope, q_rope_reshaped)

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            q *= llama_4_scaling

        if k_rope is not None:
            k = torch.cat([k, k_rope], dim=-1)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)

        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        # Cast q, k, v to fp8 for FP8 prefill path.. This will induce flashinfer to use fp8 kernels.
        out_dtype = q.dtype
        if _HAS_FP8_CONVERT_KERNEL:
            q, k, v, _, _ = mla_context_fp8_quantize(q, k, v, compute_scales=False)
        else:
            q = q.to(torch.float8_e4m3fn)
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        output_shape = (q.shape[0], layer.tp_q_head_num, layer.v_head_dim)
        if forward_batch.attn_attend_prefix_cache:
            # MHA for chunked prefix kv cache when running model with MLA
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert q_rope is None
            assert k_rope is None
            chunk_idx = forward_batch.prefix_chunk_idx

            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q,
                key=k,
                value=v,
                workspace_buffer=self.workspace_buffer,
                seq_lens=forward_batch.prefix_chunk_seq_lens[chunk_idx],
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
                o_sf_scale=-1.0,
                batch_size=forward_batch.batch_size,
                window_left=-1,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                enable_pdl=False,
                is_causal=False,
                return_lse=True,
                out=torch.zeros(*output_shape, dtype=out_dtype, device=q.device),
            )

        return flashinfer.prefill.trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self.workspace_buffer,
            seq_lens=self.forward_prefill_metadata.seq_lens,
            max_q_len=self.forward_prefill_metadata.max_seq_len,
            max_kv_len=self.forward_prefill_metadata.max_seq_len,
            bmm1_scale=layer.scaling,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=forward_batch.batch_size,
            window_left=-1,
            cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
            cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,
            enable_pdl=False,
            is_causal=True,
            return_lse=forward_batch.mha_return_lse,
            out=torch.zeros(*output_shape, dtype=out_dtype, device=q.device),
        )
