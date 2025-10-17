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
        if server_args.speculative_draft_load_format is not None:
            server_args.load_format = server_args.speculative_draft_load_format
            logger.info(
                f"Using draft model load_format: '{server_args.speculative_draft_load_format}'"
            )

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
                from sglang.private.speculative.eagle_worker import EAGLEWorker
                from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

                WorkerClass = EAGLEWorkerV2 if self.enable_overlap else EAGLEWorker

                self.draft_worker = WorkerClass(
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

    def update_cache_from_scheduler(self, schedule_batch, batch_result):
        if self.server_args.enable_suffix_decoding:
            self.tp_worker.model_runner.update_suffix_cache_from_scheduler(
                schedule_batch,
                batch_result.next_token_ids,
                batch_result.accept_length,
            )
