import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.private.speculative import spec_info as _private_spec_info  # noqa: F401
from sglang.srt.managers.scheduler import (
    GenerationBatchResult as SGLANG_GenerationBatchResult,
)
from sglang.srt.managers.scheduler import Scheduler as SGLANG_Scheduler

logger = logging.getLogger(__name__)


@dataclass
class GenerationBatchResult(SGLANG_GenerationBatchResult):
    accept_length: Optional[List[int]] = None


class Scheduler(SGLANG_Scheduler):
    def update_cache_from_scheduler(self, schedule_batch, batch_result):
        if self.server_args.enable_suffix_decoding:
            self.tp_worker.model_runner.update_suffix_cache_from_scheduler(
                schedule_batch,
                batch_result.next_token_ids,
                batch_result.accept_length,
            )
