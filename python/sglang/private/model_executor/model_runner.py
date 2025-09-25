import logging

import torch

from sglang.srt.model_executor.model_runner import ModelRunner as SGLANG_ModelRunner
from sglang.srt.model_executor.model_runner import add_mla_attention_backend
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

add_mla_attention_backend("trtllm_mla_tgl")


class ModelRunner(SGLANG_ModelRunner):

    def __init__(self, *args, server_args: ServerArgs, **kwargs):
        super().__init__(*args, server_args=server_args, **kwargs)

        # Initialize suffix cache if enabled
        self.suffix_cache = None
        if server_args.enable_suffix_decoding:
            logger.info(
                f"Initializing background batch suffix tree cache {server_args.suffix_cache_max_depth}, {server_args.suffix_cache_ratio}"
            )
            try:
                from tore_tree import SuffixCache

                self.suffix_cache = SuffixCache(
                    max_depth=server_args.suffix_cache_max_depth,
                    ratio=server_args.suffix_cache_ratio,
                )
            except (ModuleNotFoundError, ImportError):
                raise RuntimeError(
                    "\n=== Missing dependency: tore_tree ===\n"
                    "This feature requires the `tore-tree` library.\n\n"
                    "Please install it with one of the following commands:\n"
                    "for a local clone:\n"
                    "  git clone git@github.com:togethercomputer/tore-tree.git\n"
                    "  cd tore-tree\n"
                    "  pip install -e .\n"
                    "======================================\n"
                )

    def generate_suffix_draft_tokens(self, schedule_batch, last_token_ids) -> list:
        """Generate draft tokens using suffix tree speculation."""

        if self.suffix_cache is None or not hasattr(schedule_batch, "reqs"):
            return []

        # Only do suffix tree speculation during decode mode
        if (
            hasattr(schedule_batch, "forward_mode")
            and not schedule_batch.forward_mode.is_decode()
        ):
            return []
   
        results = []
        for req_idx, req in enumerate(schedule_batch.reqs):
            req_id = req.rid
            # print(f"DEBUG: Processing req {req_idx}, rid={req_id}, output_len={len(req.output_ids)}, origin_len={len(req.origin_input_ids)}")
            
            # Check if prompt is cached, if not cache it first
            has_cached = self.suffix_cache.has_cached_prompt(req_id)
            # print(f"DEBUG: req {req_id} has_cached_prompt={has_cached}")
            
            if not has_cached:
                # Cache the prompt with dummy probabilities
                prompt_token_ids = req.origin_input_ids
                prompt_probs = [1.0] * len(prompt_token_ids)
                self.suffix_cache.cache_prompt(req_id, prompt_token_ids, prompt_probs)
                # print(f"DEBUG: Cached new prompt for req {req_id}, prompt_len={len(prompt_token_ids)}")

            # Build pattern from recent tokens - FIX: use the correct last_token for this request
            max_depth = self.server_args.suffix_cache_max_depth

            # Get the most recent verified token for THIS specific request
            if req_idx < len(last_token_ids):
                last_token_for_req = [last_token_ids[req_idx]]
            else:
                last_token_for_req = []

            # Build the full token sequence: origin + output + latest verified token
            prev_tokens = req.origin_input_ids + req.output_ids + last_token_for_req
            recent_tokens = (
                prev_tokens[-max_depth:]
                if len(prev_tokens) > max_depth
                else prev_tokens
            )
            pattern = (
                recent_tokens.tolist()
                if hasattr(recent_tokens, "tolist")
                else list(recent_tokens)
            )
            # print(f"DEBUG: req {req_idx} pattern length={len(pattern)}, last_token={last_token_for_req}")

            # Speculate tokens
            max_spec_tokens = min(
                max_depth, self.model_config.context_len - len(prev_tokens) - 1
            )

            result = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=self.server_args.suffix_max_spec_factor,
                max_spec_offset=self.server_args.suffix_max_spec_offset,
                min_token_prob=self.server_args.suffix_min_token_prob,
                use_cached_prompt=False,
            )
            # print(f"DEBUG: req_idx={req_idx}, req_id={req_id}, pattern_len={len(pattern)}, result.score={getattr(result, 'score', 'N/A')}, result.token_ids={getattr(result, 'token_ids', 'N/A')[:5]}...")
            results.append(result)

        return results

    def update_suffix_cache_from_scheduler(
        self, schedule_batch, next_token_ids, accept_length_per_req_cpu
    ):
        """
        Basic suffix cache update using the core SuffixTree API.
        """
        if self.suffix_cache is None or not hasattr(schedule_batch, "reqs"):
            return

        num_reqs = min(len(schedule_batch.reqs), len(accept_length_per_req_cpu))
        if num_reqs == 0:
            return

        # Convert tokens to list
        if isinstance(next_token_ids, torch.Tensor):
            token_list = next_token_ids.cpu().tolist()
        else:
            token_list = list(next_token_ids)

        # Process each request using basic extend API
        token_idx = 0
        for req_idx, req in enumerate(schedule_batch.reqs[:num_reqs]):
            req_id = req.rid
            accept_len = accept_length_per_req_cpu[req_idx]
            total_tokens = accept_len + 1  # accepted + bonus

            # Get tokens for this request
            end_idx = min(token_idx + total_tokens, len(token_list))
            req_tokens = token_list[token_idx:end_idx]
            token_idx = end_idx

            # Cache prompt if not already cached
            if not self.suffix_cache.has_cached_prompt(req_id):
                prompt_token_ids = req.origin_input_ids
                prompt_probs = [1.0] * len(prompt_token_ids)
                self.suffix_cache.cache_prompt(req_id, prompt_token_ids, prompt_probs)

            # Update suffix cache with tokens
            if len(req_tokens) > 0:
                req_probs = [1.0] * len(req_tokens)
                self.suffix_cache.update_response(req_id, req_tokens, req_probs)

    def cleanup_finished_requests_suffix_cache(self, finished_request_ids):
        """Clean up suffix cache for finished requests to prevent memory leaks and state corruption."""
        if self.suffix_cache is None:
            return
        
        for req_id in finished_request_ids:
            try:
                if self.suffix_cache.has_cached_prompt(req_id):
                    self.suffix_cache.evict_prompt(req_id)
                # Note: Global suffix tree entries are kept for future speculation
            except ValueError:
                # Request may have already been evicted
                pass
