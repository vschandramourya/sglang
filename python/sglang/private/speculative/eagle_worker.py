import torch

from sglang.private.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.speculative.eagle_utils import assign_draft_cache_locs
from sglang.srt.speculative.eagle_worker import EAGLEWorker as SGLANG_EAGLEWorker
from sglang.srt.speculative.eagle_worker import (
    get_last_loc_large_page_size_large_top_k,
    get_last_loc_large_page_size_top_k_1,
)
from sglang.srt.utils import next_power_of_2


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

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        if self.page_size == 1:
            out_cache_loc, token_to_kv_pool_state_backup = batch.alloc_token_slots(
                num_seqs * self.speculative_num_steps * self.topk, backup_state=True
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                # TODO(lmzheng): The current implementation is still a fake support
                # for page size > 1. In the `assign_draft_cache_locs` below,
                # we directly move the indices instead of the real kv cache.
                # This only works when the kernel backend runs with page size = 1.
                # If the kernel backend runs with page size > 1, we need to
                # duplicate the real KV cache. The overhead of duplicating KV
                # cache seems okay because the draft KV cache only has one layer.
                # see a related copy operation in MHATokenToKVPool::move_kv_cache.

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )

                # TODO(lmzheng): remove this device sync
                extend_num_tokens = torch.sum(self.extend_lens).item()
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens = prefix_lens_cpu % self.page_size
                num_new_pages_per_topk = (
                    last_page_lens + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )

            out_cache_loc, token_to_kv_pool_state_backup = (
                batch.alloc_paged_token_slots_extend(
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
        )

        if self.page_size > 1 and self.topk > 1:
            # Remove padded slots
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)
