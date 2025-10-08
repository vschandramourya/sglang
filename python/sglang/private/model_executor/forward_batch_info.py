from dataclasses import dataclass
from sglang.srt.model_executor.forward_batch_info import ForwardBatchOutput as SGLANG_ForwardBatchOutput
from typing import List, Optional

@dataclass
class ForwardBatchOutput(SGLANG_ForwardBatchOutput):
    accept_length: Optional[List[int]] = None

