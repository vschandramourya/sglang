import dataclasses
import logging

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
