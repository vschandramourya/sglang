import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import EmbeddingBatchResult
from sglang.srt.managers.scheduler import (
    GenerationBatchResult as SGLANG_GenerationBatchResult,
)
from sglang.srt.managers.scheduler import Scheduler as SGLANG_Scheduler

logger = logging.getLogger(__name__)


@dataclass
class GenerationBatchResult(SGLANG_GenerationBatchResult):
    accept_length: Optional[List[int]] = None


class Scheduler(SGLANG_Scheduler):

    def launch_draft_worker(
        self, gpu_id, tp_rank, moe_ep_rank, server_args, port_args, dp_rank
    ):
        if self.spec_algorithm.is_eagle():
            if self.spec_algorithm.is_phoenix():
                from sglang.private.speculative.phoenix_worker import PhoenixWorker

                self.draft_worker = PhoenixWorker(
                    gpu_id=gpu_id,
                    tp_rank=tp_rank,
                    moe_ep_rank=moe_ep_rank,
                    server_args=server_args,
                    nccl_port=port_args.nccl_port,
                    target_worker=self.tp_worker,
                    dp_rank=dp_rank,
                )
            else:
                from sglang.srt.speculative.eagle_worker import EAGLEWorker

                self.draft_worker = EAGLEWorker(
                    gpu_id=gpu_id,
                    tp_rank=tp_rank,
                    moe_ep_rank=moe_ep_rank,
                    server_args=server_args,
                    nccl_port=port_args.nccl_port,
                    target_worker=self.tp_worker,
                    dp_rank=dp_rank,
                )
        elif self.spec_algorithm.is_standalone():
            from sglang.srt.speculative.standalone_worker import StandaloneWorker

            self.draft_worker = StandaloneWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        elif self.spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            self.draft_worker = NGRAMWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        else:
            self.draft_worker = None

    def run_batch(
        self, batch: ScheduleBatch
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Run forward
        if self.is_generation:

            batch_or_worker_batch = batch

            if self.spec_algorithm.is_none():
                # FIXME(lsyin): remove this if and finally unify the abstraction
                batch_or_worker_batch = batch.get_model_worker_batch()

            if self.enable_overlap:
                # FIXME: remove this assert
                assert isinstance(batch_or_worker_batch, ModelWorkerBatch)
                model_worker_batch = batch_or_worker_batch
                self.record_batch_in_overlap(model_worker_batch)

                # Sampling info will be modified during forward
                model_worker_batch.sampling_info = (
                    model_worker_batch.sampling_info.copy_for_forward()
                )

                bs = len(model_worker_batch.seq_lens)
                future_indices = self.future_map.alloc_future_indices(bs)

                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.default_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    if batch.sampling_info.grammars is not None:
                        model_worker_batch.delay_sample_launch = True
                    batch_result = self.model_worker.forward_batch_generation(
                        batch_or_worker_batch
                    )
                    # FIXME(lsyin): maybe move this to forward_batch_generation
                    batch_result.copy_done = torch.get_device_module(
                        self.device
                    ).Event()
                    if not model_worker_batch.delay_sample_launch:
                        self.future_map.store_to_map(
                            future_indices, batch_result.next_token_ids
                        )
                        batch_result.copy_to_cpu()
                    else:
                        batch_result.future_indices = future_indices

                # FIXME(lsyin): move this assignment elsewhere
                maybe_future_next_token_ids = -future_indices.indices
            else:
                batch_result = self.model_worker.forward_batch_generation(
                    batch_or_worker_batch
                )
                maybe_future_next_token_ids = batch_result.next_token_ids
                if self.server_args.enable_suffix_decoding:
                    self.tp_worker.model_runner.update_suffix_cache_from_scheduler(
                        batch,
                        batch_result.next_token_ids,
                        batch_result.accept_length_per_req_cpu,
                    )

            if not self.spec_algorithm.is_none():
                # TODO(lsyin): unify this metric-updating logic with non-spec, and move it to decode processing
                self.update_spec_metrics(
                    batch.batch_size(), batch_result.num_accepted_tokens
                )

            # NOTE: maybe_future_next_token_ids is used in ScheduleBatch,
            #       which can probably be replaced by future_indices later [TODO(lsyin)].
            #       we shall still keep the original outputs, e.g. next_token_ids
            #       in the GenerationBatchOutput for processing after copy_done.
            batch.output_ids = maybe_future_next_token_ids

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob or self.spec_algorithm.is_eagle():
                extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
            else:
                extend_input_len_per_req = None

            if batch.return_logprob:
                extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                extend_logprob_start_len_per_req = None

            batch_result.extend_input_len_per_req = extend_input_len_per_req
            batch_result.extend_logprob_start_len_per_req = (
                extend_logprob_start_len_per_req
            )
            return batch_result
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = EmbeddingBatchResult(embeddings=embeddings)
        return ret
