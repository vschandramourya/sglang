import faulthandler
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from http import HTTPStatus
from types import SimpleNamespace
from typing import Deque, Dict, List, Optional, Tuple, Union

import psutil
import setproctitle
import torch
import zmq
from torch.distributed import barrier

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    create_grammar_backend,
)
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    DestroyWeightsUpdateGroupReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadReqOutput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    MultiTokenizerRegisterReq,
    MultiTokenizerWrapper,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.mm_utils import init_embedding_cache
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    Req,
    RequestStage,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler import Scheduler as SGLANG_Scheduler
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,
    SchedulerMetricsMixin,
)
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_set_proc_propagate_context,
    trace_set_thread_info,
    trace_slice_batch,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_gc_logger,
    configure_logger,
    disable_request_logging,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    get_zmq_socket,
    kill_itself_when_parent_died,
    numa_bind_to_node,
    point_to_point_pyobj,
    pyspy_dump_schedulers,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback
from sglang.srt.managers.scheduler import GenerationBatchResult as SGLANG_GenerationBatchResult
from sglang.srt.managers.scheduler import EmbeddingBatchResult

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
                        batch, batch_result.next_token_ids, batch_result.accept_length_per_req_cpu
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

