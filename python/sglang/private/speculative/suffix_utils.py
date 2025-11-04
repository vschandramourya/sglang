from typing import List, Tuple

import torch

from sglang.srt.speculative.eagle_info import EagleDraftInput


def build_suffix_tree_draft_lists(
    suffix_spec_tokens_batch: List[List[int]],
    batch_size: int,
    spec_info: EagleDraftInput,
    speculative_num_steps,
    device,
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

    score_list: List[torch.Tensor] = []
    token_list: List[torch.Tensor] = []
    parents_list: List[torch.Tensor] = []

    for step in range(speculative_num_steps):
        if step == 0:
            # Step 0: shape (b, 1, 1), (b, 1), (b, 2)
            # Use topk_p and topk_index from spec_info (from previous round's draft model)
            scores = spec_info.topk_p.unsqueeze(1)  # (b, 1, 1)
            tokens = spec_info.topk_index  # (b, 1)
            # Parents: [-1, 0] for each request
            parents = torch.tensor([[-1, 0]], dtype=torch.int64, device=device).repeat(
                batch_size, 1
            )

            score_list.append(scores)
            token_list.append(tokens)
            parents_list.append(parents)
        else:
            # Step i>0: shape (b, 1, 1), (b, 1), (b, 1)
            # Use suffix tree tokens: step 1 uses suffix_tokens[0], step 2 uses suffix_tokens[1], etc.
            scores = torch.ones(batch_size, 1, 1, dtype=torch.float32, device=device)
            tokens = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
            # Parent index: topk² * (step - 1) + topk = 1 * (step - 1) + 1 = step
            parents = torch.full(
                (batch_size, 1), step, dtype=torch.int64, device=device
            )

            # Populate with suffix tree tokens (step-1 because step 0 uses topk_index)
            for req_idx in range(batch_size):
                suffix_tokens = suffix_spec_tokens_batch[req_idx]
                tokens[req_idx, 0] = suffix_tokens[step - 1]

            score_list.append(scores)
            token_list.append(tokens)
            parents_list.append(parents)

    return score_list, token_list, parents_list


def apply_suffix_tree_tokens(draft_tokens, suffix_spec_tokens_batch):
    """
    Apply suffix tree tokens to CUDA-graph output (post-organize) in place.

    Mutates only `draft_tokens`: keeps column 0 (verifier) untouched and writes
    suffix tokens into columns [1 .. N-1], truncating if suffix is longer.
    """

    if not isinstance(suffix_spec_tokens_batch, list) or not suffix_spec_tokens_batch:
        return  # in-place; nothing to do

    B, N = draft_tokens.shape  # N == num_draft_token - 1

    # iterate over whichever is smaller to avoid IndexError if suffix list < B
    for b in range(min(B, len(suffix_spec_tokens_batch))):
        suffix_toks = suffix_spec_tokens_batch[b]
        if not suffix_toks:
            continue

        # map suffix[0] -> col 1, suffix[1] -> col 2, ... ; truncate if longer
        max_write = min(len(suffix_toks), N - 1)
        for j in range(max_write):
            draft_tokens[b, 1 + j] = int(suffix_toks[j])
