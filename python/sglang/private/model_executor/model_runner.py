import logging

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

logger = logging.getLogger(__name__)


_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()


class ModelRunner(SGLANG_ModelRunner):
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
                    "fa4",
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
