import os
from typing import Optional

import torch

from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.layers.sampler import (
    top_k_top_p_min_p_sampling_from_probs_torch as SGLANG_top_k_top_p_min_p_sampling_from_probs_torch,
)

_PRIVATE_SAMPLER_ENV = "TGL_USE_PRIVATE_SAMPLER"


def _should_use_private_sampler() -> bool:
    """Return True if the private sampler should be enabled via env flag."""
    return os.getenv(_PRIVATE_SAMPLER_ENV, "0") == "1"


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):
    """
    A top-k, top-p and min-p sampling implementation with native pytorch operations.
    When sampling_seed is not None, deterministic inference will be enabled, it will sample
    with the sampling_seed of each request.
    """
    if not _should_use_private_sampler():
        return SGLANG_top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            top_ks,
            top_ps,
            min_ps,
            need_min_p_sampling,
            sampling_seed,
            positions,
        )

    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    probs_sort = torch.nan_to_num(probs_sort, nan=0.0, posinf=0.0, neginf=0.0)
    probs_sort = probs_sort.clamp_min_(0.0)

    row_sum = probs_sort.sum(dim=1)
    zero_row = row_sum <= 0

    if zero_row.any():
        probs_sort = probs_sort.clone()
        probs_sort[zero_row] = 0.0
        probs_sort[zero_row, 0] = 1.0

    if sampling_seed is not None:
        sampled_index = multinomial_with_seed(probs_sort, sampling_seed, positions)
    else:
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids
