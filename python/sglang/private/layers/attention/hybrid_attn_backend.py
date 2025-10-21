from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import (
    HybridAttnBackend as HybridAttnBackendSrt,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class HybridAttnBackend(HybridAttnBackendSrt):
    def _select_backend(self, forward_mode: ForwardMode) -> AttentionBackend:
        """
        Select the appropriate attention backend based on the forward mode.

        Args:
            forward_mode: The current forward mode indicating the operation type

        Returns:
            The selected attention backend (prefill or decode)

        Note:
            - decode_or_idle: Always uses decode backend
            - target_verify or draft_extend: Uses decode backend if speculative_attention_mode is "decode", otherwise prefill backend
            - prefill: Always uses prefill backend
        """
        if forward_mode.is_decode_or_idle():
            return self.decode_backend
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend(
            include_v2=True
        ):
            return (
                self.decode_backend
                if self.model_runner.server_args.speculative_attention_mode == "decode"
                else self.prefill_backend
            )
        else:
            return self.prefill_backend
