from typing import List, Tuple

import torch

from sglang.private.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.eagle_worker import EAGLEWorker as SGLANG_EAGLEWorker
from sglang.srt.speculative.spec_utils import fast_topk, select_top_k_tokens


class EAGLEWorker(SGLANG_EAGLEWorker):

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(
                batch
            )
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                self.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids, seq_lens_cpu
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
                accept_length=[],
            )
        else:
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                spec_info = self.draft(batch)
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )

            with self.draft_tp_context(self.draft_model_runner.tp_group):
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                if (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                ):
                    # decode is not finished
                    self.forward_draft_extend_after_decode(batch)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                can_run_cuda_graph=can_run_cuda_graph,
                accept_length=verify_output.accept_length_per_req_cpu,
            )

    def _build_suffix_tree_draft_lists(
        self,
        suffix_spec_tokens_batch: List[List[int]],
        batch_size: int,
        spec_info: EagleDraftInput,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Build score_list, token_list, parents_list from suffix tree tokens.

        This creates the same structure as eagle/phoenix draft to ensure compatibility
        with verification and cuda graph.

        Assumes topk=1 (greedy decoding only).

        IMPORTANT: Step 0 uses spec_info.topk_index (draft model prediction),
        NOT suffix tokens! Suffix tokens are used from step 1 onwards.

        For topk=1:
        - Step 0: scores (b,1,1), tokens (b,1)=topk_index, parents (b,2) = [-1, 0]
        - Step i>0: scores (b,1,1), tokens (b,1)=suffix_tokens[i-1], parents (b,1)

        Args:
            suffix_spec_tokens_batch: List of token lists, one per request in batch
            batch_size: Number of requests in batch
            spec_info: EagleDraftInput containing topk_index for step 0

        Returns:
            Tuple of (score_list, token_list, parents_list) with length speculative_num_steps
        """
        assert self.topk == 1, "Suffix tree optimization currently only supports topk=1"

        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        for step in range(self.speculative_num_steps):
            if step == 0:
                # Step 0: shape (b, 1, 1), (b, 1), (b, 2)
                # Use topk_p and topk_index from spec_info (from previous round's draft model)
                scores = spec_info.topk_p.unsqueeze(1)  # (b, 1, 1)
                tokens = spec_info.topk_index  # (b, 1)
                # Parents: [-1, 0] for each request
                parents = torch.tensor(
                    [[-1, 0]], dtype=torch.int64, device=self.device
                ).repeat(batch_size, 1)

                score_list.append(scores)
                token_list.append(tokens)
                parents_list.append(parents)
            else:
                # Step i>0: shape (b, 1, 1), (b, 1), (b, 1)
                # Use suffix tree tokens: step 1 uses suffix_tokens[0], step 2 uses suffix_tokens[1], etc.
                scores = torch.ones(
                    batch_size, 1, 1, dtype=torch.float32, device=self.device
                )
                tokens = torch.zeros(
                    batch_size, 1, dtype=torch.int64, device=self.device
                )
                # Parent index: topk² * (step - 1) + topk = 1 * (step - 1) + 1 = step
                parents = torch.full(
                    (batch_size, 1), step, dtype=torch.int64, device=self.device
                )

                # Populate with suffix tree tokens (step-1 because step 0 uses topk_index)
                for req_idx in range(batch_size):
                    suffix_tokens = suffix_spec_tokens_batch[req_idx]
                    tokens[req_idx, 0] = suffix_tokens[step - 1]

                score_list.append(scores)
                token_list.append(tokens)
                parents_list.append(parents)

        return score_list, token_list, parents_list

    def draft(self, batch: ScheduleBatch):
        # Parse args
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_batch = self.topk
        spec_info.num_tokens_for_logprob_per_batch = self.topk
        batch.return_hidden_states = False

        # Check if we should use suffix tree tokens - handle multiple requests in batch
        suffix_spec_tokens_batch = None
        all_requests_use_suffix_tree = False
        if self.server_args.enable_suffix_decoding and hasattr(
            self.target_worker, "model_runner"
        ):
            try:
                # Get the tokens that Phoenix will use as starting points for each request
                last_token_ids = []
                batch_size = spec_info.topk_index.shape[0]
                # Collect first token for each request in the batch
                for req_idx in range(batch_size):
                    phoenix_first_token = spec_info.topk_index[req_idx][0].item()
                    last_token_ids.append(phoenix_first_token)
                # Generate suffix tree proposals for all requests in batch
                suffix_draft_results = (
                    self.target_worker.model_runner.generate_suffix_draft_tokens(
                        batch, last_token_ids
                    )
                )
                if suffix_draft_results:
                    min_score = (
                        self.server_args.suffix_min_score_ratio
                        * self.speculative_num_draft_tokens
                    )
                    suffix_spec_tokens_batch = []
                    # Process each request's suffix tree result separately
                    # Todo: improvement can be support suffix tree batch processing
                    for batch_idx, result in enumerate(suffix_draft_results):
                        if batch_idx < batch_size:  # Ensure we don't exceed batch size
                            if hasattr(result, "score") and hasattr(
                                result, "token_ids"
                            ):
                                if (
                                    result.score >= min_score
                                    and len(result.token_ids) > 0
                                ):
                                    # score is high enough, use suffix tree tokens
                                    suffix_spec_tokens_batch.append(result.token_ids)
                                else:
                                    # score is too low, still use phoenix generated tokens
                                    suffix_spec_tokens_batch.append(None)
                            else:
                                suffix_spec_tokens_batch.append(None)
                    # If no request has valid suffix tree tokens, set to None
                    if not any(tokens for tokens in suffix_spec_tokens_batch):
                        suffix_spec_tokens_batch = None
                    else:
                        # Check if ALL requests have valid suffix tree tokens with sufficient length
                        # We need speculative_num_steps - 1 tokens because step 0 uses topk_index
                        all_requests_use_suffix_tree = all(
                            tokens is not None for tokens in suffix_spec_tokens_batch
                        )
            except Exception as e:
                print(f"Suffix tree generation failed: {e}")
                suffix_spec_tokens_batch = None

        # OPTIMIZATION: Skip phoenix/eagle draft if all requests use suffix tree
        if all_requests_use_suffix_tree:
            # Fast path: use suffix tree tokens directly without running draft model
            # Build score_list, token_list, parents_list with same structure as eagle/phoenix
            score_list, token_list, parents_list = self._build_suffix_tree_draft_lists(
                suffix_spec_tokens_batch, batch_size, spec_info
            )
            parent_list, top_scores_index, draft_tokens = organize_draft_results(
                score_list, token_list, parents_list, self.speculative_num_draft_tokens
            )
            seq_lens_sum = batch.seq_lens_sum
            seq_lens_cpu = batch.seq_lens.cpu()
        else:
            # Normal path: run phoenix/eagle draft
            # Get forward batch
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.suffix_spec_tokens = suffix_spec_tokens_batch
            assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            forward_batch.suffix_spec_tokens = suffix_spec_tokens_batch
            can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
                forward_batch
            )
            if can_cuda_graph:
                parent_list, top_scores_index, draft_tokens = (
                    self.cuda_graph_runner.replay(forward_batch)
                )
            else:
                forward_batch.can_run_dp_cuda_graph = False
                if not forward_batch.forward_mode.is_idle():
                    # Initialize attention backend
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                # Run forward steps
                parent_list, top_scores_index, draft_tokens = self.draft_forward(
                    forward_batch
                )
            seq_lens_sum = forward_batch.seq_lens_sum
            seq_lens_cpu = forward_batch.seq_lens_cpu

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        suffix_spec_tokens_batch = getattr(forward_batch, "suffix_spec_tokens", None)

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )

            # If we have suffix tree tokens, replace the selected tokens with suffix tree tokens for each request
            if (
                suffix_spec_tokens_batch
                and isinstance(suffix_spec_tokens_batch, list)
                and i > 0
            ):
                step_idx = i - 1  # Step index for suffix tokens (0-indexed)
                batch_size = tree_info[1].shape[0]
                # Process each request in the batch
                for req_idx in range(min(batch_size, len(suffix_spec_tokens_batch))):
                    if suffix_spec_tokens_batch[req_idx] is not None:
                        req_suffix_tokens = suffix_spec_tokens_batch[req_idx]
                        if isinstance(req_suffix_tokens, list) and step_idx < len(
                            req_suffix_tokens
                        ):
                            # Use suffix tree token for this request at this step
                            suffix_token = req_suffix_tokens[step_idx]
                            tree_info[1][req_idx].fill_(
                                suffix_token
                            )  # Replace token for this request
                            tree_info[0][req_idx].fill_(
                                1.0
                            )  # Suffix tree only do greedy pick now
                            input_ids[req_idx] = (
                                suffix_token  # Update input_id for this request
                            )

            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            # This is a temporary fix for the case that the user is using standalone
            # speculative decoding and the draft model architecture is gpt-oss. gpt-oss
            # rope kernel needs cache_loc to be contiguous.
            if (
                self.server_args.speculative_algorithm == "STANDALONE"
                and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output, _ = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            )
            self._detect_nan_if_needed(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )
        return parent_list, top_scores_index, draft_tokens
