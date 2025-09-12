import dataclasses
import logging
import os
import random

from sglang.srt.hf_transformers_utils import check_gguf_file
from sglang.srt.server_args import ServerArgs as SGLANG_ServerArgs
from sglang.srt.server_args import add_attention_backend_choices
from sglang.srt.utils import (
    get_device,
    get_device_memory_capacity,
    is_flashinfer_available,
    is_hip,
    is_remote_url,
    is_sm100_supported,
)
from sglang.utils import is_in_ci

logger = logging.getLogger(__name__)

TGL_PRIVATE_ATTENTION_BACKENDS = [
    "trtllm_mla_tgl",
]

add_attention_backend_choices(TGL_PRIVATE_ATTENTION_BACKENDS)


@dataclasses.dataclass
class ServerArgs(SGLANG_ServerArgs):

    def __post_init__(self):
        # Check deprecated arguments
        if self.enable_ep_moe:
            self.ep_size = self.tp_size
            print_deprecated_warning(
                "NOTE: --enable-ep-moe is deprecated. Please set `--ep-size` to the same value as `--tp-size` instead."
            )
        if self.enable_deepep_moe:
            self.moe_a2a_backend = "deepep"
            print_deprecated_warning(
                "NOTE: --enable-deepep-moe is deprecated. Please set `--moe-a2a-backend` to 'deepep' instead."
            )
        if self.enable_triton_kernel_moe:
            self.moe_runner_backend = "triton_kernel"
            print_deprecated_warning(
                "NOTE: --enable-triton-kernel-moe is deprecated. Please set `--moe-runner-backend` to 'triton_kernel' instead."
            )
        if self.enable_flashinfer_cutlass_moe:
            self.moe_runner_backend = "flashinfer_cutlass"
            print_deprecated_warning(
                "NOTE: --enable-flashinfer-cutlass-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutlass' instead."
            )
        if self.enable_flashinfer_trtllm_moe:
            self.moe_runner_backend = "flashinfer_trtllm"
            print_deprecated_warning(
                "NOTE: --enable-flashinfer-trtllm-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_trtllm' instead."
            )
        if self.enable_flashinfer_mxfp4_moe:
            self.moe_runner_backend = "flashinfer_mxfp4"
            print_deprecated_warning(
                "NOTE: --enable-flashinfer-mxfp4-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_mxfp4' instead."
            )

        # Set missing default values
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.served_model_name is None:
            self.served_model_name = self.model_path
        if self.device is None:
            self.device = get_device()
        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)

        gpu_mem = get_device_memory_capacity(self.device)

        # Set mem fraction static
        if self.mem_fraction_static is None:
            if gpu_mem is not None:
                # GPU memory capacity = model weights + KV cache pool + activations + cuda graph buffers
                # mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity.

                # We want mem_fraction_static to be as large as possible but still has enough room
                # for activations and cuda graph buffers. We use the following heuristic to
                # compute the needed size for activations and cuda graph buffers:
                # - The size of the activation depends on the chunked_prefill_size and model size.
                # - The size of cuda graph buffers depends on the cuda graph capture range and model size.
                # For GPUs with more memory, we use a larger chunked_prefill_size and
                # capture more cuda graphs, so they need to reserve more memory.
                parallel_size = self.tp_size * self.pp_size

                if gpu_mem < 20 * 1024:
                    # T4, 4080. (chunked_prefill_size 2k, cuda_graph_max_bs 8)
                    reserved_mem = (2.8 + parallel_size / 10) * 1024
                elif gpu_mem < 35 * 1024:
                    # A10, L40, 4090, 5090. (chunked_prefill_size 2k, cuda_graph_max_bs 8)
                    reserved_mem = (2.8 + parallel_size / 10) * 1024
                elif gpu_mem < 90 * 1024:
                    # H100, A100. (chunked_prefill_size 8k, cuda_graph_max_bs 160)
                    reserved_mem = (9.5 + parallel_size / 2) * 1024
                elif gpu_mem < 100 * 1024:
                    # H20. (chunked_prefill_size 8k, cuda_graph_max_bs 256)
                    reserved_mem = (12 + parallel_size / 2) * 1024
                elif gpu_mem < 160 * 1024:
                    # H200. (chunked_prefill_size 8k, cuda_graph_max_bs 256)
                    reserved_mem = (12 + parallel_size / 2) * 1024
                else:
                    # B200, MI300. (chunked_prefill_size 16k, cuda_graph_max_bs 512)
                    reserved_mem = 32 * 1024

                # draft model and larger cuda graph buffers
                if self.speculative_algorithm is not None:
                    if self.speculative_algorithm == "STANDALONE":
                        # Standalone speculative decoding needs more memory than other speculative
                        # decoding algorithms since the draft model is typically larger.
                        reserved_mem += 6 * 1024
                    else:
                        reserved_mem += 2 * 1024
                if self.enable_dp_attention:
                    reserved_mem += 4 * 1024

                self.mem_fraction_static = round((gpu_mem - reserved_mem) / gpu_mem, 3)
            else:
                self.mem_fraction_static = 0.88

            # Lazy init to avoid circular import
            # Multimodal models need more memory for the image processor
            from sglang.srt.configs.model_config import ModelConfig

            model_config = ModelConfig.from_server_args(self)
            if model_config.is_multimodal:
                self.adjust_mem_fraction_for_vlm(model_config)

        # Set chunked prefill size, which depends on the gpu memory capacity
        if self.chunked_prefill_size is None:
            if gpu_mem is not None:
                if gpu_mem < 35 * 1024:  # A10, L40, 4090
                    self.chunked_prefill_size = 2048
                elif gpu_mem < 160 * 1024:  # H100, H200, A100, H20
                    self.chunked_prefill_size = 8192
                else:  # B200, MI300
                    self.chunked_prefill_size = 16384
            else:
                self.chunked_prefill_size = 4096

        # Set cuda graph max batch size
        if self.cuda_graph_max_bs is None:
            # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM<25G, you can either disable cuda graph or set `cuda_graph_max_bs` to a very small value to reduce the memory overhead of creating cuda graphs, with almost no impact on performance. However, when serving models with TP4 or TP8, we need to enable cuda graph to maintain high performance. In this case, we can set `cuda_graph_max_bs` to 80 (half of the default value 160) to reduce the memory overhead of creating cuda graphs. Looking at the logs from TP4 serving of qwen2-72b, a value of 80 is sufficient and can reduce the memory overhead of creating cuda graphs on lower-end GPUs compared to the original 160, avoiding OOM issues.
            if gpu_mem is not None and gpu_mem < 35 * 1024:
                if self.tp_size < 4:
                    self.cuda_graph_max_bs = 8
                else:
                    self.cuda_graph_max_bs = 80

        # Set kernel backends for hpu device
        if self.device == "hpu":
            self.attention_backend = "torch_native"
            self.sampling_backend = "pytorch"

        # Model-specific adjustments
        self.model_specific_adjustments()

        # Set kernel backends
        if self.device == "cpu":
            if self.attention_backend is None:
                self.attention_backend = "intel_amx"
            self.sampling_backend = "pytorch"

        if self.sampling_backend is None:
            self.sampling_backend = (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )

        if self.attention_backend == "torch_native":
            logger.warning(
                "Cuda graph is disabled because of using torch native attention backend"
            )
            self.disable_cuda_graph = True

        if self.attention_backend == "ascend":
            logger.warning(
                "At this moment Ascend attention backend only supports a page_size of 128, change page_size to 128."
            )
            self.page_size = 128

        if (
            self.attention_backend == "flashmla"
            or self.decode_attention_backend == "flashmla"
        ):
            logger.warning(
                "FlashMLA only supports a page_size of 64, change page_size to 64."
            )
            self.page_size = 64

        if (
            self.attention_backend == "cutlass_mla"
            or self.decode_attention_backend == "cutlass_mla"
        ):
            logger.warning(
                "Cutlass MLA only supports a page_size of 128, change page_size to 128."
            )
            self.page_size = 128

        if (
            self.attention_backend == "trtllm_mla"
            or self.decode_attention_backend == "trtllm_mla"
        ):
            if not is_sm100_supported():
                raise ValueError(
                    "TRTLLM MLA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
                )

            if self.page_size not in [32, 64]:
                logger.warning(
                    f"TensorRT-LLM MLA only supports page_size of 32 or 64, changing page_size from {self.page_size} to 64."
                )
                self.page_size = 64

            if self.kv_cache_dtype not in ["fp8_e4m3", "auto"]:
                raise ValueError(
                    "TensorRT-LLM MLA backend only supports kv-cache-dtype of fp8_e4m3 or auto."
                )

        if (
            self.attention_backend == "trtllm_mla_tgl"
            or self.decode_attention_backend == "trtllm_mla_tgl"
        ):
            if not is_sm100_supported():
                raise ValueError(
                    "TRTLLM MLA TGL backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
                )

            if self.page_size not in [32, 64]:
                logger.warning(
                    f"TensorRT-LLM MLA TGL only supports page_size of 32 or 64, changing page_size from {self.page_size} to 64."
                )
                self.page_size = 64

            if self.kv_cache_dtype not in ["fp8_e4m3", "auto"]:
                raise ValueError(
                    "TensorRT-LLM MLA TGL backend only supports kv-cache-dtype of fp8_e4m3 or auto."
                )

        if (
            self.attention_backend == "trtllm_mha"
            or self.decode_attention_backend == "trtllm_mha"
            or self.prefill_attention_backend == "trtllm_mha"
        ):
            if not is_sm100_supported():
                raise ValueError(
                    "TRTLLM MHA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
                )

            if self.page_size not in [16, 32, 64]:
                logger.warning(
                    f"TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing page_size from {self.page_size} to 64."
                )
                self.page_size = 64

        if self.attention_backend == "dual_chunk_flash_attn":
            logger.warning(
                "Mixed chunk, radix cache, and cuda graphs are disabled because of using dual chunk flash attention backend"
            )
            self.enable_mixed_chunk = False
            self.disable_cuda_graph = True
            self.disable_radix_cache = True

        # Set page size
        if self.page_size is None:
            self.page_size = 1

        # AMD-specific Triton attention KV splits default number
        if is_hip():
            self.triton_attention_num_kv_splits = 16

        # Choose grammar backend
        if self.grammar_backend is None:
            self.grammar_backend = "xgrammar"

        if self.dp_size == 1:
            self.enable_dp_attention = False

        # Data parallelism attention
        if self.enable_dp_attention:
            self.schedule_conservativeness = self.schedule_conservativeness * 0.3
            assert self.tp_size % self.dp_size == 0
            self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
            logger.warning(
                f"DP attention is enabled. The chunked prefill size is adjusted to {self.chunked_prefill_size} to avoid MoE kernel issues. "
            )

        if self.enable_dp_lm_head:
            assert (
                self.enable_dp_attention
            ), "Please enable dp attention when setting enable_dp_lm_head. "

        # MoE kernel
        if self.moe_runner_backend == "flashinfer_cutlass":
            assert (
                self.quantization == "modelopt_fp4"
            ), "modelopt_fp4 quantization is required for Flashinfer MOE"
            assert self.ep_size in [
                1,
                self.tp_size,
            ], "The expert parallel size must be 1 or the same as the tensor parallel size"

        if self.moe_runner_backend == "flashinfer_trtllm":
            assert (
                self.quantization == "modelopt_fp4" or self.quantization == "fp8"
            ), "modelopt_fp4 quantization is required for Flashinfer TRTLLM MoE"
            self.disable_shared_experts_fusion = True
            logger.warning(
                "FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set."
            )

        # DeepEP MoE
        if self.moe_a2a_backend == "deepep":
            if self.deepep_mode == "normal":
                logger.warning("Cuda graph is disabled because deepep_mode=`normal`")
                self.disable_cuda_graph = True
            self.ep_size = self.tp_size
            logger.warning(
                f"DeepEP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        if self.enable_eplb and (self.expert_distribution_recorder_mode is None):
            self.expert_distribution_recorder_mode = "stat"
            logger.warning(
                "EPLB is enabled. The expert_distribution_recorder_mode is automatically set."
            )

        if (self.enable_eplb or (self.init_expert_location is not None)) and (
            self.ep_dispatch_algorithm is None
        ):
            self.ep_dispatch_algorithm = "static"

        if self.enable_eplb:
            assert self.ep_size > 1

        if self.enable_expert_distribution_metrics and (
            self.expert_distribution_recorder_mode is None
        ):
            self.expert_distribution_recorder_mode = "stat"

        if self.expert_distribution_recorder_buffer_size is None:
            if (x := self.eplb_rebalance_num_iterations) is not None:
                self.expert_distribution_recorder_buffer_size = x
            elif self.expert_distribution_recorder_mode is not None:
                self.expert_distribution_recorder_buffer_size = 1000

        # Pipeline parallelism
        if self.pp_size > 1:
            self.disable_overlap_schedule = True
            logger.warning(
                "Pipeline parallelism is incompatible with overlap schedule."
            )

        # Hicache
        if self.hicache_storage_backend == "mooncake":
            # to use mooncake storage backend, the following conditions must be met:
            self.hicache_io_backend = "kernel"
            self.hicache_mem_layout = "page_first"

        # Speculative Decoding
        if self.speculative_algorithm == "NEXTN":
            # NEXTN shares the same implementation of EAGLE
            self.speculative_algorithm = "EAGLE"

        if self.speculative_algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
            if self.speculative_algorithm == "STANDALONE":
                # TODO: support dp attention for standalone speculative decoding
                assert (
                    self.enable_dp_attention is False
                ), "Currently standalone speculative decoding does not support dp attention."
            if self.max_running_requests is None:
                self.max_running_requests = 48
            self.disable_overlap_schedule = True
            logger.warning(
                "Overlap scheduler is disabled because of using "
                "eagle speculative decoding."
            )
            if self.enable_mixed_chunk:
                self.enable_mixed_chunk = False
                logger.warning(
                    "Mixed chunked prefill is disabled because of using "
                    "eagle speculative decoding."
                )

            model_arch = self.get_hf_config().architectures[0]
            if model_arch in ["DeepseekV3ForCausalLM", "Glm4MoeForCausalLM"]:
                # Auto set draft_model_path DeepSeek-V3/R1
                if self.speculative_draft_model_path is None:
                    self.speculative_draft_model_path = self.model_path
                else:
                    logger.warning(
                        "DeepSeek MTP does not require setting speculative_draft_model_path."
                    )

            # Auto choose parameters
            if self.speculative_num_steps is None:
                assert (
                    self.speculative_eagle_topk is None
                    and self.speculative_num_draft_tokens is None
                )
                (
                    self.speculative_num_steps,
                    self.speculative_eagle_topk,
                    self.speculative_num_draft_tokens,
                ) = auto_choose_speculative_params(self)

            if (
                self.attention_backend == "trtllm_mha"
                or self.decode_attention_backend == "trtllm_mha"
                or self.prefill_attention_backend == "trtllm_mha"
            ):
                if self.speculative_eagle_topk > 1:
                    raise ValueError(
                        "trtllm_mha backend only supports topk = 1 for speculative decoding."
                    )

            if (
                self.speculative_eagle_topk == 1
                and self.speculative_num_draft_tokens != self.speculative_num_steps + 1
            ):
                logger.warning(
                    "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
                )
                self.speculative_num_draft_tokens = self.speculative_num_steps + 1

            if (
                self.speculative_eagle_topk > 1
                and self.page_size > 1
                and self.attention_backend != "flashinfer"
            ):
                raise ValueError(
                    "speculative_eagle_topk > 1 with page_size > 1 is unstable and produces incorrect results for paged attention backends. This combination is only supported for the 'flashinfer' backend."
                )

            # The token generated from the verify step is counted.
            # If sepculative_num_steps >= speculative_num_draft_tokens, the additional tokens will definitely be discarded.
            # assert self.speculative_num_steps < self.speculative_num_draft_tokens

        # GGUF
        if (
            self.load_format == "auto" or self.load_format == "gguf"
        ) and check_gguf_file(self.model_path):
            self.quantization = self.load_format = "gguf"

        # Model loading
        if is_remote_url(self.model_path):
            self.load_format = "remote"
        if self.custom_weight_loader is None:
            self.custom_weight_loader = []

        # PD disaggregation
        if self.disaggregation_mode == "decode":
            assert (
                self.disaggregation_decode_tp is None
            ), "Cannot set --disaggregation-decode-tp for the decode engine."
            assert (
                self.disaggregation_decode_dp is None
            ), "Cannot set --disaggregation-decode-dp for the decode engine."

            self.disable_radix_cache = True
            logger.warning("KV cache is forced as chunk cache for decode server")

            if self.dp_size > 1 and not is_in_ci():
                assert self.prefill_round_robin_balance, (
                    "Prefill round robin balance is required when dp size > 1. "
                    "Please make sure that the prefill instance is launched with `--load-balance-method round_robin`"
                    " and `--prefill-round-robin-balance` is set for decode server."
                )
        elif self.disaggregation_mode == "prefill":
            if self.disaggregation_decode_tp is None:
                self.disaggregation_decode_tp = self.tp_size
            if self.disaggregation_decode_dp is None:
                self.disaggregation_decode_dp = self.dp_size

            self.disaggregation_prefill_pp = self.pp_size
            self.validate_disagg_tp_size(self.tp_size, self.disaggregation_decode_tp)

            self.disable_cuda_graph = True
            logger.warning("Cuda graph is disabled for prefill server")

        # Propagate env vars
        os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = (
            "1" if self.enable_torch_compile else "0"
        )
        # Set env var before grammar backends init
        os.environ["SGLANG_DISABLE_OUTLINES_DISK_CACHE"] = (
            "1" if self.disable_outlines_disk_cache else "0"
        )

        if self.enable_hierarchical_cache and self.disable_radix_cache:
            raise ValueError(
                "The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive "
                "and cannot be used at the same time. Please use only one of them."
            )
