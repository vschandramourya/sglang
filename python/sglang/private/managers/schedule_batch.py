import dataclasses
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch as SGLangScheduleBatch
from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode


@dataclasses.dataclass
class ScheduleBatch(SGLangScheduleBatch):
    
    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        bs = len(self.reqs)

        if self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone() or self.spec_algorithm.is_phoenix():
            # if spec decoding is used, the decode batch is prepared inside
            # `forward_batch_speculative_generation` after running draft models.
            return

        if self.sampling_info.penalizer_orchestrator.is_required:
            if self.enable_overlap:
                # TODO: this can be slow, optimize this.
                delayed_output_ids = torch.tensor(
                    [
                        (
                            req.output_ids[-1]
                            if len(req.output_ids)
                            else req.origin_input_ids[-1]
                        )
                        for req in self.reqs
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    delayed_output_ids
                )
            else:
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    self.output_ids.to(torch.int64)
                )

        # Update fields
        self.input_ids = self.output_ids
        self.output_ids = None

        if self.model_config.is_encoder_decoder:
            locs = self.encoder_lens + self.seq_lens
            self.prepare_encoder_info_decode()
        else:
            locs = self.seq_lens.clone()

        if self.enable_overlap:
            # Do not use in-place operations in the overlap mode
            self.seq_lens = self.seq_lens + 1
            self.orig_seq_lens = self.orig_seq_lens + 1
        else:
            # A faster in-place version
            self.seq_lens.add_(1)
            self.orig_seq_lens.add_(1)
        self.seq_lens_sum += bs

        # free memory
        if isinstance(self.tree_cache, SWAChunkCache):
            for req in self.reqs:
                self.tree_cache.evict_swa(
                    req, req.seqlen - 1, self.model_config.attention_chunk_size
                )

        # Allocate memory
        if self.token_to_kv_pool_allocator.page_size == 1:
            self.out_cache_loc = self.alloc_token_slots(bs)
        else:
            last_loc = self.req_to_token_pool.req_to_token[
                self.req_pool_indices, self.seq_lens - 2
            ]
            self.out_cache_loc = self.alloc_paged_token_slots_decode(
                self.seq_lens, last_loc
            )

        self.req_to_token_pool.write(
            (self.req_pool_indices, locs), self.out_cache_loc.to(torch.int32)
        )
