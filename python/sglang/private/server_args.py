import argparse
import dataclasses
import logging

from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES
from sglang.srt.server_args import ServerArgs as SGLANG_ServerArgs
from sglang.srt.server_args import add_attention_backend_choices
from sglang.srt.utils import is_sm100_supported

logger = logging.getLogger(__name__)

TGL_PRIVATE_ATTENTION_BACKENDS = [
    "trtllm_mla_tgl",
]

add_attention_backend_choices(TGL_PRIVATE_ATTENTION_BACKENDS)


@dataclasses.dataclass
class ServerArgs(SGLANG_ServerArgs):

    # Draft model attention backend
    draft_attention_backend: str = None

    # Suffix tree decoding
    enable_suffix_decoding: bool = False
    suffix_cache_max_depth: int = 64
    suffix_prompt_cutoff_length: int = 128
    suffix_max_spec_factor: float = 1.0
    suffix_max_spec_offset: float = 0.0
    suffix_min_token_prob: float = 0.1
    suffix_min_score_ratio: float = 1.2

    def _handle_other_validations(self):
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
