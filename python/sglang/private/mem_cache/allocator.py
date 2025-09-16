import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator as SGLangPagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator import alloc_extend_kernel
from sglang.srt.utils import next_power_of_2


class PagedTokenToKVPoolAllocator(SGLangPagedTokenToKVPoolAllocator):

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        self.seen_max_num_extend_tokens_next_power_of_2 = max(
            self.seen_max_num_extend_tokens_next_power_of_2,
            next_power_of_2(extend_num_tokens),
        )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.ret_values,
            next_power_of_2(bs),
            self.page_size,
            self.seen_max_num_extend_tokens_next_power_of_2,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        # avoid item() sync overhead by using cpu tensors
        num_pages_after = (seq_lens_cpu + self.page_size - 1) // self.page_size
        num_pages_before = (prefix_lens_cpu + self.page_size - 1) // self.page_size
        num_new_pages = num_pages_after - num_pages_before
        extend_lens = seq_lens_cpu - prefix_lens_cpu
        sum_num_new_pages = torch.sum(num_new_pages).to(torch.int64)
        merged_value = (sum_num_new_pages) << 32 | torch.sum(extend_lens).to(
            torch.int64
        )

        merged_value = merged_value.item()
        num_new_pages = merged_value >> 32
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
