import logging
from typing import Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner as SGLANG_ModelRunner
from sglang.srt.model_executor.model_runner import (
    cpu_has_amx_support,
    is_fa3_default_architecture,
    is_flashinfer_available,
    is_hip,
    is_hopper_with_cuda_12_3,
    is_no_spec_infer_or_topk_one,
    is_npu,
    is_sm100_supported,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()


class ModelRunner(SGLANG_ModelRunner):

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ):
        super().__init__(
            model_config,
            mem_fraction_static,
            gpu_id,
            tp_rank,
            tp_size,
            moe_ep_rank,
            moe_ep_size,
            pp_rank,
            pp_size,
            nccl_port,
            server_args,
            dp_rank,
            is_draft_worker,
            req_to_token_pool,
            token_to_kv_pool_allocator,
        )

        # Initialize suffix cache if enabled
        self.suffix_cache = None
        if server_args.enable_suffix_decoding:
            logger.info(
                f"Initializing background batch suffix tree cache {server_args.suffix_cache_max_depth}, {server_args.suffix_cache_ratio}"
            )
            try:
                from tore_tree import SuffixCache

                self.suffix_cache = SuffixCache(
                    max_depth=server_args.suffix_cache_max_depth,
                    ratio=server_args.suffix_cache_ratio,
                )
            except ImportError as e:
                print(
                    "Error: tore_tree is not installed. Please checkout https://github.com/togethercomputer/tore-tree and install it with: pip install -e ."
                )
                raise e

    def model_specific_adjustment(self):
        server_args = self.server_args

        if (
            server_args.attention_backend == "intel_amx"
            and server_args.device == "cpu"
            and not _is_cpu_amx_available
        ):
            logger.info(
                "The current platform does not support Intel AMX, will fallback to torch_native backend."
            )
            server_args.attention_backend = "torch_native"

        if server_args.prefill_attention_backend is not None and (
            server_args.prefill_attention_backend
            == server_args.decode_attention_backend
        ):  # override the default attention backend
            server_args.attention_backend = server_args.prefill_attention_backend

        if (
            getattr(self.model_config.hf_config, "dual_chunk_attention_config", None)
            is not None
        ):
            if server_args.attention_backend is None:
                server_args.attention_backend = "dual_chunk_flash_attn"
                logger.info("Dual chunk attention is turned on by default.")
            elif server_args.attention_backend != "dual_chunk_flash_attn":
                raise ValueError(
                    "Dual chunk attention is enabled, but attention backend is set to "
                    f"{server_args.attention_backend}. Please set it to 'dual_chunk_flash_attn'."
                )

        if server_args.attention_backend is None:
            """
            Auto select the fastest attention backend.

            1. Models with MHA Architecture (e.g: Llama, QWen)
                1.1 We will turn on FA3 on hopper unless user use spec decode with topk > 1 or page_size > 1.
                1.2 In other cases, we will use flashinfer if available, otherwise use triton.
            2. Models with MLA Architecture and using FA3
                2.1 We will use FA3 backend on hopper.
                2.2 We will use Flashinfer backend on blackwell.
                2.3 Otherwise, we will use triton backend.
            """

            if not self.use_mla_backend:
                # MHA architecture
                if (
                    is_hopper_with_cuda_12_3()
                    and is_no_spec_infer_or_topk_one(server_args)
                    and is_fa3_default_architecture(self.model_config.hf_config)
                ):
                    server_args.attention_backend = "fa3"
                elif _is_hip:
                    server_args.attention_backend = "aiter"
                elif _is_npu:
                    server_args.attention_backend = "ascend"
                else:
                    server_args.attention_backend = (
                        "flashinfer" if is_flashinfer_available() else "triton"
                    )
            else:
                # MLA architecture
                if is_hopper_with_cuda_12_3():
                    server_args.attention_backend = "fa3"
                elif is_sm100_supported():
                    server_args.attention_backend = "flashinfer"
                elif _is_hip:
                    head_num = self.model_config.get_num_kv_heads(self.tp_size)
                    # TODO current aiter only support head number 16 or 128 head number
                    if (
                        head_num == 128 or head_num == 16
                    ) and self.spec_algorithm.is_none():
                        server_args.attention_backend = "aiter"
                    else:
                        server_args.attention_backend = "triton"
                elif _is_npu:
                    server_args.attention_backend = "ascend"
                else:
                    server_args.attention_backend = "triton"
            logger.info(
                f"Attention backend not explicitly specified. Use {server_args.attention_backend} backend by default."
            )
        elif self.use_mla_backend:
            if server_args.device != "cpu":
                if server_args.attention_backend in [
                    "aiter",
                    "flashinfer",
                    "fa3",
                    "triton",
                    "flashmla",
                    "cutlass_mla",
                    "trtllm_mla",
                    "trtllm_mla_tgl",
                    "ascend",
                ]:
                    logger.info(
                        f"MLA optimization is turned on. Use {server_args.attention_backend} backend."
                    )
                else:
                    raise ValueError(
                        f"Invalid attention backend for MLA: {server_args.attention_backend}"
                    )
            else:
                if server_args.attention_backend != "intel_amx":
                    raise ValueError(
                        "MLA optimization not supported on CPU except for intel_amx backend."
                    )

        if (
            server_args.attention_backend == "fa3"
            and server_args.kv_cache_dtype == "fp8_e5m2"
        ):
            logger.warning(
                "FlashAttention3 only supports fp8_e4m3 if using FP8; "
                "Setting attention backend to triton."
            )
            server_args.attention_backend = "triton"

        if server_args.enable_double_sparsity:
            logger.info(
                "Double sparsity optimization is turned on. Use triton backend without CUDA graph."
            )
            server_args.attention_backend = "triton"
            server_args.disable_cuda_graph = True

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if not self.use_mla_backend:
            server_args.disable_chunked_prefix_cache = True
        # TODO(kaixih@nvidia): remove this once we have a better solution for DP attention.
        #  For more details, see: https://github.com/sgl-project/sglang/issues/8616
        elif (
            self.dp_size > 1
            and is_sm100_supported()
            and server_args.attention_backend != "triton"
        ):
            logger.info(
                "Disable chunked prefix cache when dp size > 1 and attention backend is not triton."
            )
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            logger.info("Chunked prefix cache is turned on.")

        if server_args.attention_backend == "aiter":
            if self.model_config.context_len > 8192:
                self.mem_fraction_static *= 0.85

        if (
            server_args.enable_hierarchical_cache
            and server_args.hicache_io_backend == "kernel"
        ):
            # fix for the compatibility issue with FlashAttention3 decoding and HiCache kernel backend
            if server_args.decode_attention_backend is None:
                if not self.use_mla_backend:
                    server_args.decode_attention_backend = (
                        "flashinfer" if is_flashinfer_available() else "triton"
                    )
                else:
                    server_args.decode_attention_backend = (
                        "flashinfer" if is_sm100_supported() else "triton"
                    )
            elif server_args.decode_attention_backend == "fa3":
                server_args.hicache_io_backend = "direct"
                logger.warning(
                    "FlashAttention3 decode backend is not compatible with hierarchical cache. "
                    f"Setting hicache_io_backend to vanilla I/O, which may lead to suboptimal performance with small page sizes."
                )

    def _get_attention_backend_from_str(self, backend_str: str):
        if backend_str == "flashinfer":
            if not self.use_mla_backend:
                from sglang.srt.layers.attention.flashinfer_backend import (
                    FlashInferAttnBackend,
                )

                # Init streams
                if (
                    self.server_args.speculative_algorithm == "EAGLE"
                    or self.server_args.speculative_algorithm == "PHOENIX"
                ):
                    if (
                        not hasattr(self, "plan_stream_for_flashinfer")
                        or not self.plan_stream_for_flashinfer
                    ):
                        self.plan_stream_for_flashinfer = torch.cuda.Stream()
                return FlashInferAttnBackend(self)
            else:
                from sglang.srt.layers.attention.flashinfer_mla_backend import (
                    FlashInferMLAAttnBackend,
                )

                return FlashInferMLAAttnBackend(self)
        elif backend_str == "aiter":
            from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

            return AiterAttnBackend(self)
        elif self.server_args.attention_backend == "wave":
            from sglang.srt.layers.attention.wave_backend import WaveAttnBackend

            return WaveAttnBackend(self)
        elif backend_str == "ascend":
            from sglang.srt.layers.attention.ascend_backend import AscendAttnBackend

            return AscendAttnBackend(self)
        elif backend_str == "triton":
            assert not self.model_config.is_encoder_decoder, (
                "Cross attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            if self.server_args.enable_double_sparsity:
                from sglang.srt.layers.attention.double_sparsity_backend import (
                    DoubleSparseAttnBackend,
                )

                return DoubleSparseAttnBackend(self)
            else:
                from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

                return TritonAttnBackend(self)
        elif backend_str == "torch_native":
            from sglang.srt.layers.attention.torch_native_backend import (
                TorchNativeAttnBackend,
            )

            return TorchNativeAttnBackend(self)
        elif backend_str == "flashmla":
            from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

            return FlashMLABackend(self)
        elif backend_str == "fa3":
            assert (
                torch.cuda.get_device_capability()[0] == 8 and not self.use_mla_backend
            ) or torch.cuda.get_device_capability()[0] == 9, (
                "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
                "Please use `--attention-backend flashinfer`."
            )
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )

            return FlashAttentionBackend(self)
        elif backend_str == "cutlass_mla":
            from sglang.srt.layers.attention.cutlass_mla_backend import (
                CutlassMLABackend,
            )

            return CutlassMLABackend(self)
        elif backend_str == "trtllm_mla":
            if not self.use_mla_backend:
                raise ValueError("trtllm_mla backend can only be used with MLA models.")
            from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

            return TRTLLMMLABackend(self)
        elif backend_str == "trtllm_mla_tgl":
            if not self.use_mla_backend:
                raise ValueError(
                    "trtllm_mla_tgl backend can only be used with MLA models."
                )
            from sglang.private.layers.attention.trtllm_mla_tgl_backend import (
                TRTLLMMLATGLBackend,
            )

            return TRTLLMMLATGLBackend(self)
        elif backend_str == "trtllm_mha":
            if self.use_mla_backend:
                raise ValueError(
                    "trtllm_mha backend can only be used with non-MLA models."
                )
            from sglang.srt.layers.attention.trtllm_mha_backend import (
                TRTLLMHAAttnBackend,
            )

            return TRTLLMHAAttnBackend(self)
        elif backend_str == "intel_amx":
            from sglang.srt.layers.attention.intel_amx_backend import (
                IntelAMXAttnBackend,
            )

            return IntelAMXAttnBackend(self)
        elif backend_str == "dual_chunk_flash_attn":
            from sglang.srt.layers.attention.dual_chunk_flashattention_backend import (
                DualChunkFlashAttentionBackend,
            )

            return DualChunkFlashAttentionBackend(self)
        else:
            raise ValueError(f"Invalid attention backend: {backend_str}")

    def generate_suffix_draft_tokens(self, schedule_batch, last_token_ids) -> list:
        """Generate draft tokens using suffix tree speculation."""

        if self.suffix_cache is None or not hasattr(schedule_batch, "reqs"):
            return []

        # Only do suffix tree speculation during decode mode
        if (
            hasattr(schedule_batch, "forward_mode")
            and not schedule_batch.forward_mode.is_decode()
        ):
            return []

        results = []
        for req_idx, req in enumerate(schedule_batch.reqs):
            req_id = req.rid

            # Check if prompt is cached, if not cache it first
            if not self.suffix_cache.has_cached_prompt(req_id):
                # Cache the prompt with dummy probabilities
                prompt_token_ids = req.origin_input_ids
                prompt_probs = [1.0] * len(prompt_token_ids)
                self.suffix_cache.cache_prompt(req_id, prompt_token_ids, prompt_probs)

            # Build pattern from recent tokens - FIX: use the correct last_token for this request
            max_depth = self.server_args.suffix_cache_max_depth

            # Get the most recent verified token for THIS specific request
            if req_idx < len(last_token_ids):
                last_token_for_req = [last_token_ids[req_idx]]
            else:
                last_token_for_req = []

            # Build the full token sequence: origin + output + latest verified token
            prev_tokens = req.origin_input_ids + req.output_ids + last_token_for_req
            recent_tokens = (
                prev_tokens[-max_depth:]
                if len(prev_tokens) > max_depth
                else prev_tokens
            )
            pattern = (
                recent_tokens.tolist()
                if hasattr(recent_tokens, "tolist")
                else list(recent_tokens)
            )

            # Speculate tokens
            max_spec_tokens = min(
                max_depth, self.model_config.context_len - len(prev_tokens) - 1
            )

            result = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=self.server_args.suffix_max_spec_factor,
                max_spec_offset=self.server_args.suffix_max_spec_offset,
                min_token_prob=self.server_args.suffix_min_token_prob,
            )
            results.append(result)

        return results

    def update_suffix_cache_from_scheduler(
        self, schedule_batch, next_token_ids, accept_length_per_req_cpu
    ):
        """
        Basic suffix cache update using the core SuffixTree API.
        """
        if self.suffix_cache is None or not hasattr(schedule_batch, "reqs"):
            return

        num_reqs = min(len(schedule_batch.reqs), len(accept_length_per_req_cpu))
        if num_reqs == 0:
            return

        # Convert tokens to list
        if isinstance(next_token_ids, torch.Tensor):
            token_list = next_token_ids.cpu().tolist()
        else:
            token_list = list(next_token_ids)

        # Process each request using basic extend API
        token_idx = 0
        for req_idx, req in enumerate(schedule_batch.reqs[:num_reqs]):
            req_id = req.rid
            accept_len = accept_length_per_req_cpu[req_idx]
            total_tokens = accept_len + 1  # accepted + bonus

            # Get tokens for this request
            end_idx = min(token_idx + total_tokens, len(token_list))
            req_tokens = token_list[token_idx:end_idx]
            token_idx = end_idx

            # Cache prompt if not already cached
            if not self.suffix_cache.has_cached_prompt(req_id):
                prompt_token_ids = req.origin_input_ids
                prompt_probs = [1.0] * len(prompt_token_ids)
                self.suffix_cache.cache_prompt(req_id, prompt_token_ids, prompt_probs)

            # Update suffix cache with tokens
            if len(req_tokens) > 0:
                req_probs = [1.0] * len(req_tokens)
                self.suffix_cache.update_response(req_id, req_tokens, req_probs)
