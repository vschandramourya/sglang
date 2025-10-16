from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner as SGLANG_EAGLEDraftCudaGraphRunner,
)


class SGLANG_EAGLEDraftCudaGraphRunner(SGLANG_EAGLEDraftCudaGraphRunner):

    def _apply_suffix_tree_tokens_to_cuda_graph_output(
        self, out, suffix_spec_tokens_batch, raw_bs
    ):
        """
        Apply suffix tree tokens to CUDA-graph output (post-organize) in place.

        Mutates only `draft_tokens`: keeps column 0 (verifier) untouched and writes
        suffix tokens into columns [1 .. N-1], truncating if suffix is longer.
        Does NOT modify `top_scores_index`.
        """
        parent_list, top_scores_index, draft_tokens = out

        if (
            not isinstance(suffix_spec_tokens_batch, list)
            or not suffix_spec_tokens_batch
        ):
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

    def replay(self, forward_batch: ForwardBatch):
        out = super().replay(forward_batch)
        raw_bs = forward_batch.batch_size

        # Apply suffix tree tokens if present
        suffix_spec_tokens_batch = getattr(forward_batch, "suffix_spec_tokens", None)
        if suffix_spec_tokens_batch:
            self._apply_suffix_tree_tokens_to_cuda_graph_output(
                out, suffix_spec_tokens_batch, raw_bs
            )

        return out
