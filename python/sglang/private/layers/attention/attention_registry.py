from sglang.srt.layers.attention.attention_registry import register_attention_backend


@register_attention_backend("trtllm_mla_tgl")
def create_trtllm_mla_tgl_backend(runner):
    if not runner.use_mla_backend:
        raise ValueError("trtllm_mla_tgl backend can only be used with MLA models.")
    from sglang.private.layers.attention.trtllm_mla_tgl_backend import (
        TRTLLMMLATGLBackend,
    )

    return TRTLLMMLATGLBackend(runner)
