from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.speculative.eagle_worker import EAGLEWorker as SGLANG_EAGLEWorker


class EAGLEWorker(SGLANG_EAGLEWorker):

    def _create_decode_backend(self):
        backend_map = {
            "flashinfer": self._create_flashinfer_decode_backend,
            "triton": self._create_triton_decode_backend,
            "aiter": self._create_aiter_decode_backend,
            "fa3": self._create_fa3_decode_backend,
            "flashmla": self._create_flashmla_decode_backend,
            "trtllm_mha": self._create_trtllm_mha_decode_backend,
            "trtllm_mla": self._create_trtllm_mla_decode_backend,
            "trtllm_mla_tgl": self._create_trtllm_mla_tgl_decode_backend,
        }

        return self._create_backend(
            "decode_attention_backend",
            backend_map,
            "EAGLE is not supported in decode attention backend {backend_type}",
        )

    def _create_draft_extend_backend(self):
        backend_map = {
            "flashinfer": self._create_flashinfer_prefill_backend,
            "triton": self._create_triton_prefill_backend,
            "aiter": self._create_aiter_prefill_backend,
            "fa3": self._create_fa3_prefill_backend,
            "trtllm_mha": self._create_trtllm_mha_prefill_backend,
            "trtllm_mla": self._create_trtllm_mla_prefill_backend,
            "trtllm_mla_tgl": self._create_trtllm_mla_tgl_prefill_backend,
        }
        backend_name = (
            "decode_attention_backend"
            if self.server_args.speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        return self._create_backend(
            backend_name,
            backend_map,
            "EAGLE is not supported in attention backend {backend_type}",
        )

    def _create_trtllm_mla_tgl_decode_backend(self):
        if not global_server_args_dict["use_mla_backend"]:
            raise ValueError(
                "trtllm_mla_tgl  backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.private.layers.attention.trtllm_mla_tgl_backend import (
            TRTLLMMLATGLMultiStepDraftBackend,
        )

        self.has_prefill_wrapper_verify = True
        return TRTLLMMLATGLMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mla_tgl_prefill_backend(self):
        if not global_server_args_dict["use_mla_backend"]:
            raise ValueError(
                "trtllm_mla_tgl backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.private.layers.attention.trtllm_mla_tgl_backend import (
            TRTLLMMLATGLBackend,
        )

        return TRTLLMMLATGLBackend(self.draft_model_runner, skip_prefill=False)
