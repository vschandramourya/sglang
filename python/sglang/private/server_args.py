import argparse
import dataclasses
import logging
import os

from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES
from sglang.srt.server_args import ServerArgs as SGLANG_ServerArgs
from sglang.srt.server_args import (
    add_attention_backend_choices,
    auto_choose_speculative_params,
)
from sglang.srt.utils import get_bool_env_var, is_sm100_supported

logger = logging.getLogger(__name__)

TGL_PRIVATE_ATTENTION_BACKENDS = [
    "trtllm_mla_tgl",
]

FLASHINFER_FP4_GEMM_BACKENDS = ["cudnn", "trtllm", "cutlass"]
FP4_GEMM_BACKEND_CHOICES = FLASHINFER_FP4_GEMM_BACKENDS + ["sglang"]

add_attention_backend_choices(TGL_PRIVATE_ATTENTION_BACKENDS)


@dataclasses.dataclass
class ServerArgs(SGLANG_ServerArgs):

    # Draft model attention backend
    draft_attention_backend: str = None

    # FP4 GEMM backend
    fp4_gemm_backend: str = "cutlass"

    # Suffix tree decoding
    enable_suffix_decoding: bool = False
    suffix_cache_max_depth: int = 64
    suffix_prompt_cutoff_length: int = 128
    suffix_max_spec_factor: float = 1.0
    suffix_max_spec_offset: float = 0.0
    suffix_min_token_prob: float = 0.1
    suffix_min_score_ratio: float = 1.2

    # Incremental tokenizer Related
    enable_inc_tokenizer: bool = False
    verify_inc_tokenization_correctness: bool = False
    enable_trtllm_mla_fp8_prefill: bool = False

    def _handle_other_validations(self):
        # Validate FP4 GEMM backend consistency
        if (
            get_bool_env_var("SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM", "false")
            and self.fp4_gemm_backend != "cutlass"
        ):
            raise ValueError(
                f"Conflicting FP4 GEMM backend configuration: "
                f"SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM=true but --fp4-gemm-backend={self.fp4_gemm_backend}. "
                f"Please either unset the environment variable or use --fp4-gemm-backend=cutlass."
            )

        # Check for reverse conflict: cutlass flag with explicitly false env var
        if (
            self.fp4_gemm_backend == "cutlass"
            and "SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM" in os.environ
            and not get_bool_env_var("SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM")
        ):
            raise ValueError(
                f"Conflicting FP4 GEMM backend configuration: "
                f"--fp4-gemm-backend=cutlass but SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM={os.environ.get('SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM')}. "
                f"Please either unset the environment variable or use a different --fp4-gemm-backend value."
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

        if self.enable_trtllm_mla_fp8_prefill and not (
            self.attention_backend == "trtllm_mla"
            or self.prefill_attention_backend == "trtllm_mla"
        ):
            raise ValueError(
                "TRTLLM MLA FP8 prefill is only supported with trtllm_mla backend."
            )

    def _handle_speculative_decoding(self):
        super()._handle_speculative_decoding()

        if self.speculative_algorithm == "PHOENIX":
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

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        SGLANG_ServerArgs.add_cli_args(parser)
        for action in parser._actions:
            if action.dest == "speculative_algorithm":
                action.choices.append("PHOENIX")

        parser.add_argument(
            "--draft-attention-backend",
            type=str,
            default=ServerArgs.draft_attention_backend,
            choices=ATTENTION_BACKEND_CHOICES,
            help="Attention backend for the draft model.",
        )

        parser.add_argument(
            "--fp4-gemm-backend",
            type=str,
            default=ServerArgs.fp4_gemm_backend,
            choices=FP4_GEMM_BACKEND_CHOICES,
            help="Backend for FP4 GEMM operations. 'cudnn' uses flashinfer's cuDNN backend, 'trtllm' uses flashinfer's TensorRT-LLM backend, 'cutlass' uses flashinfer's CUTLASS backend, 'sglang' uses SGLang's kernel backend (no flashinfer).",
        )

        # Suffix tree decoding
        parser.add_argument(
            "--enable-suffix-decoding",
            action="store_true",
            help="Enable suffix tree decoding for speculative decoding.",
        )
        parser.add_argument(
            "--suffix-cache-max-depth",
            type=int,
            default=ServerArgs.suffix_cache_max_depth,
            help="Maximum depth of the suffix tree cache.",
        )
        parser.add_argument(
            "--suffix-prompt-cutoff-length",
            type=int,
            default=ServerArgs.suffix_prompt_cutoff_length,
            help="Maximum length of prompt to cutoff.",
        )
        parser.add_argument(
            "--suffix-max-spec-factor",
            type=float,
            default=ServerArgs.suffix_max_spec_factor,
            help="Maximum speculation factor for suffix tree.",
        )
        parser.add_argument(
            "--suffix-max-spec-offset",
            type=float,
            default=ServerArgs.suffix_max_spec_offset,
            help="Maximum speculation offset for suffix tree.",
        )
        parser.add_argument(
            "--suffix-min-token-prob",
            type=float,
            default=ServerArgs.suffix_min_token_prob,
            help="Minimum token probability for suffix tree speculation.",
        )
        parser.add_argument(
            "--suffix-min-score-ratio",
            type=float,
            default=ServerArgs.suffix_min_score_ratio,
            help="Minimum score ratio for suffix tree to override other methods.",
        )
        parser.add_argument(
            "--suffix-enable-score",
            action="store_true",
            help="Enable score-based suffix tree override.",
        )
        parser.add_argument(
            "--enable-inc-tokenizer",
            action="store_true",
            help="Enable incremental tokenizer with caching.",
        )
        parser.add_argument(
            "--verify-inc-tokenization-correctness",
            action="store_true",
            help="Verify correctness of incremental tokenizer against original tokenizer.",
        )
        parser.add_argument(
            "--enable-trtllm-mla-fp8-prefill",
            action="store_true",
            help="Enable FP8 prefill attention for TRTLLM MLA backend.",
        )
