"""
Ultra-simple prefix-based tokenization cache.

This implementation stores text->tokens mappings and does simple prefix matching.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .utils import compare_tokenizations

DEBUG = False

logger = logging.getLogger(__name__)


class TokenizationCache:
    """
    Dead simple cache that just stores recent text->tokens mappings
    and checks for exact or prefix matches.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        boundary_whitelist_tokens: Optional[Set[int]] = None,
    ):
        # Store tuples of (text, kwargs_hash, tokens)
        self.cache: List[Tuple[str, str, List[int]]] = []
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self.partial_hits = 0
        self.boundary_whitelist_tokens = boundary_whitelist_tokens

        # Compile regex for finding special tokens
        self.special_token_pattern = re.compile(r"<[｜|][^｜|>]*[｜|]>")

    def _kwargs_key(self, **kwargs) -> str:
        """Create a simple key from kwargs."""
        return str(sorted(kwargs.items()))

    def get_exact(self, text: str, **kwargs) -> Optional[List[int]]:
        """Check for exact match."""
        kwargs_key = self._kwargs_key(**kwargs)

        for cached_text, cached_kwargs, tokens in self.cache:
            if cached_text == text and cached_kwargs == kwargs_key:
                self.hits += 1
                # Move to front (LRU)
                self.cache.remove((cached_text, cached_kwargs, tokens))
                self.cache.insert(0, (cached_text, cached_kwargs, tokens))
                return tokens.copy()

        return None

    def get_prefix(self, text: str, **kwargs) -> Optional[Tuple[List[int], int]]:
        """Find longest prefix match."""
        kwargs_key = self._kwargs_key(**kwargs)
        best_prefix_len = 0
        best_tokens = None

        unsafe_prefixes = []
        for cached_text, cached_kwargs, tokens in self.cache:
            if cached_kwargs == kwargs_key and text.startswith(cached_text):
                # Skip empty token lists (can't check boundary)
                if len(tokens) == 0:
                    continue

                if tokens[-1] in self.boundary_whitelist_tokens:
                    if len(cached_text) > best_prefix_len:
                        best_prefix_len = len(cached_text)
                        best_tokens = tokens
                else:
                    unsafe_prefixes.append(tokens)

        if best_tokens:
            self.partial_hits += 1
            return best_tokens.copy(), best_prefix_len
        elif len(unsafe_prefixes) > 0:
            last_tokens = [
                unsafe_prefix[-1] if len(unsafe_prefix) > 0 else None
                for unsafe_prefix in unsafe_prefixes[:5]
            ]
            print(
                f"Out of {len(unsafe_prefixes)} potential prefix matches, none with safe boundary found. Last tokens of first 5 unsafe prefixes: {last_tokens}"
            )
        else:
            print(f"Out of {len(self.cache)} cache entries, no prefix match found.")

        return None

    def put(self, text: str, tokens: List[int], **kwargs):
        """Store a text->tokens mapping exactly at special token boundaries."""
        kwargs_key = self._kwargs_key(**kwargs)

        # Default: cache the full text
        cache_text = text
        cache_tokens = tokens.copy()
        # if len(cache_tokens) > 0 and cache_tokens[-1] not in self.boundary_whitelist_tokens:
        #     if DEBUG:
        #         print(f'Last token {cache_tokens[-1]} not in boundary whitelist tokens {self.boundary_whitelist_tokens}, not caching')
        #     return

        # If we have boundary whitelist tokens, find the last safe boundary
        if self.boundary_whitelist_tokens and len(tokens) > 0:
            # Check if last token is already a special token
            if tokens[-1] not in self.boundary_whitelist_tokens:
                # Walk backwards to find the last special token
                last_special_idx = None
                for i in range(len(tokens) - 1, -1, -1):
                    if tokens[i] in self.boundary_whitelist_tokens:
                        last_special_idx = i
                        break

                if last_special_idx is not None:
                    # We found a special token at position last_special_idx
                    # Cache tokens[:last_special_idx+1] which ends at the special token
                    cache_tokens = tokens[: last_special_idx + 1]

                    # average of 4 chars per token, 20 chars per token is enough range
                    search_range = (len(tokens) - last_special_idx) * 20
                    search_text = text[-1 * search_range :]

                    all_matches = list(self.special_token_pattern.finditer(search_text))

                    # get last match
                    last_match = all_matches[-1] if len(all_matches) > 0 else None
                    if last_match is None:
                        # no special token found, return
                        logger.info(f"No special token found, returning")
                        return

                    cache_text = (
                        text[: -1 * search_range] + search_text[: last_match.end()]
                    )
                    if DEBUG:
                        print(f"cache_text[-200:]: {cache_text[-200:]}")
                else:
                    # no special token found, return
                    logger.info(f"No special token found, returning")
                    return
            else:
                if DEBUG:
                    print(
                        f"Last token {tokens[-1]} from cache_text[-200:] {cache_text[-200:]} in boundary whitelist tokens, caching"
                    )

        # Remove if already exists
        for i, (cached_text, cached_kwargs, _) in enumerate(self.cache):
            if cached_text == cache_text and cached_kwargs == kwargs_key:
                del self.cache[i]
                break

        # Add to front
        self.cache.insert(0, (cache_text, kwargs_key, cache_tokens))

        # Trim if needed
        if len(self.cache) > self.max_entries:
            self.cache = self.cache[: self.max_entries]

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.partial_hits = 0

    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.partial_hits + self.misses
        hit_rate = (self.hits + self.partial_hits) / total if total > 0 else 0

        return {
            "hits": self.hits,
            "partial_hits": self.partial_hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }


class TokenizerWrapper:
    """
    Ultra-simple wrapper that just does exact and prefix matching.
    """

    def __init__(
        self,
        tokenizer,
        max_cache_entries: int = 1000,
        enable_boundary_check: bool = True,
        verify_tokenization_correctness: bool = False,
    ):
        self.tokenizer = tokenizer

        if hasattr(tokenizer, "_original_encode"):
            self._direct_encode = tokenizer._original_encode
        else:
            self._direct_encode = tokenizer.encode

        self.enable_boundary_check = enable_boundary_check
        self.verify_tokenization_correctness = verify_tokenization_correctness

        # Initialize boundary whitelist tokens (safe boundary tokens)
        self.boundary_whitelist_tokens = self._get_default_boundary_tokens()

        self.cache = TokenizationCache(
            max_cache_entries, self.boundary_whitelist_tokens
        )

        # Statistics for boundary violations
        self.boundary_violations = 0
        self.successful_boundaries = 0

    def _get_default_boundary_tokens(self) -> Set[int]:
        """Get default safe boundary tokens from the tokenizer."""
        safe_tokens = set()

        # Tokens from dsv3-agent-mystery-2-7-online-mtp-FP4 tokenizer
        # that are not contained in any other tokens (thus safe boundaries)
        self.safe_boundaries = [
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜▁pad▁｜>",
            "<｜fim▁hole｜>",
            "<｜fim▁begin｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<|EOT|>",
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁calls▁end｜>",
            "<｜tool▁call▁begin｜>",
            "<｜tool▁call▁end｜>",
            "<｜tool▁outputs▁begin｜>",
            "<｜tool▁outputs▁end｜>",
            "<｜tool▁output▁begin｜>",
            "<｜tool▁output▁end｜>",
            "<｜tool▁sep｜>",
        ]

        # logger.debug(f"Initializing {len(self.safe_boundaries)} default boundary tokens")
        print(f"Initializing {len(self.safe_boundaries)} default boundary tokens")
        # Tokenize each safe boundary character/string
        for boundary in self.safe_boundaries:
            try:
                # Tokenize without special tokens to get the raw token ID
                tokens = self._direct_encode(boundary, add_special_tokens=False)
                assert (
                    len(tokens) == 1
                ), f"Expected 1 token, got {len(tokens)}. If you'd like to use boundary safe caching, please add the special tokens for this tokenizer."
                safe_tokens.add(tokens[0])
            except Exception as e:
                # Some tokenizers might not handle single chars well
                raise RuntimeError(
                    f"Error tokenizing safe boundary: {boundary}. Error: {e}"
                )

        # Also add BOS/EOS token IDs if available
        if (
            hasattr(self.tokenizer, "bos_token_id")
            and self.tokenizer.bos_token_id is not None
        ):
            safe_tokens.add(self.tokenizer.bos_token_id)
        if (
            hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            safe_tokens.add(self.tokenizer.eos_token_id)
        if (
            hasattr(self.tokenizer, "pad_token_id")
            and self.tokenizer.pad_token_id is not None
        ):
            safe_tokens.add(self.tokenizer.pad_token_id)

        # logger.debug(f"Initialized {len(safe_tokens)} default boundary tokens")
        print(f"Initialized {len(safe_tokens)} default boundary tokens")
        print(f"Default boundary tokens: {safe_tokens}")
        print(f"Default boundary strings: {self.safe_boundaries}")
        return safe_tokens

    def _verify_boundary(
        self, prefix_tokens: List[int], text: str, prefix_len: int
    ) -> bool:
        """
        Verify that the boundary between cached prefix and new text is safe.

        Returns True if the boundary is safe (can use cache), False otherwise.
        """
        if not self.enable_boundary_check or not self.boundary_whitelist_tokens:
            return True  # No boundary checking enabled

        # if not prefix_tokens:
        #     return True  # Empty prefix is always safe

        # Check the last token of the prefix
        if len(prefix_tokens) == 0:
            return False

        last_prefix_token = prefix_tokens[-1]

        # If we're not confident about the boundary, reject it
        if last_prefix_token not in self.boundary_whitelist_tokens:
            if DEBUG:
                print(
                    f"Last prefix token {last_prefix_token} not in boundary whitelist tokens {self.boundary_whitelist_tokens}"
                )
            return False

        if DEBUG:
            print(
                f"Last prefix token {last_prefix_token} in boundary whitelist tokens {self.boundary_whitelist_tokens}"
            )
        return True

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """
        Encode with simple caching:
        1. Check exact match
        2. Check prefix match and verify boundary safety
        3. Full tokenization if boundary check fails
        """
        start_time = time.time()
        # Try exact match
        tokens = self.cache.get_exact(
            text, add_special_tokens=add_special_tokens, **kwargs
        )
        if tokens is not None:
            cached_time = time.time() - start_time
            if self.verify_tokenization_correctness:
                self.verify(tokens, cached_time, text, add_special_tokens, kwargs)
            return tokens

        # Try prefix match
        prefix_result = self.cache.get_prefix(
            text, add_special_tokens=add_special_tokens, **kwargs
        )

        if prefix_result:
            prefix_tokens, prefix_len = prefix_result

            # Verify the boundary is safe before using the cache
            if self._verify_boundary(prefix_tokens, text, prefix_len):
                # Boundary is safe, use cached prefix
                remainder = text[prefix_len:]
                if remainder:
                    # Tokenize remainder without special tokens
                    remainder_tokens = self._direct_encode(
                        remainder, add_special_tokens=add_special_tokens, **kwargs
                    )

                    # Combine (prefix already has special tokens if needed)
                    tokens = prefix_tokens + remainder_tokens
                else:
                    tokens = prefix_tokens

                self.successful_boundaries += 1
            else:
                # Boundary verification failed, fall back to full tokenization
                self.boundary_violations += 1
                if len(prefix_tokens) > 0:
                    last_token_info = f"{prefix_tokens[-1]} ({self.tokenizer.decode([prefix_tokens[-1]])})"
                else:
                    last_token_info = "N/A (empty token list)"

                msg = f"""
                Boundary check failed at position {prefix_len} in text of length {len(text)}. Falling back to full tokenization.
                Last prefix token: {last_token_info}.
                Last prefix token whitelist: {self.boundary_whitelist_tokens} ({self.safe_boundaries}).
                Falling back to full tokenization.
                """
                print(msg)
                tokens = self._direct_encode(
                    text, add_special_tokens=add_special_tokens, **kwargs
                )
                # Don't count this as a cache miss since we found a prefix but couldn't use it
        else:
            # No prefix found, full tokenization
            tokens = self._direct_encode(
                text, add_special_tokens=add_special_tokens, **kwargs
            )
            self.cache.misses += 1

        # Cache the result
        self.cache.put(text, tokens, add_special_tokens=add_special_tokens, **kwargs)

        cache_tokenizer_time = time.time() - start_time

        if self.verify_tokenization_correctness:
            self.verify(tokens, cache_tokenizer_time, text, add_special_tokens, kwargs)

        return tokens

    def verify(
        self,
        tokens: List[int],
        cached_time: float,
        text: str,
        add_special_tokens: bool,
        kwargs: Dict[str, Any],
    ):
        start_time = time.time()
        # Reference (non-cached) tokenization
        if hasattr(self.tokenizer, "hf"):
            ref_tokens = self.tokenizer.hf.encode(
                text, add_special_tokens=add_special_tokens, **kwargs
            )
        elif hasattr(self.tokenizer, "_original_encode"):
            ref_tokens = self.tokenizer._original_encode(
                text, add_special_tokens=add_special_tokens, **kwargs
            )
        else:
            ref_tokens = self._direct_encode(
                text, add_special_tokens=add_special_tokens, **kwargs
            )
        ref_tokenizer_time = time.time() - start_time

        # Compare and log
        compare_tokenizations(
            tokens, ref_tokens, cached_time, ref_tokenizer_time, len(text)
        )

    def decode(self, *args, **kwargs):
        """Delegate to underlying tokenizer."""
        return self.tokenizer.decode(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes."""
        if "tokenizer" not in self.__dict__:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.tokenizer, name)

    @property
    def vocab_size(self):
        """Get vocab size from underlying tokenizer."""
        if hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, "get_vocab"):
            return len(self.tokenizer.get_vocab())
        else:
            raise AttributeError("Cannot determine vocab_size")

    def __len__(self):
        """Delegate to underlying tokenizer."""
        return len(self.tokenizer)

    def get_boundary_stats(self):
        """Get boundary checking statistics."""
        total_boundary_checks = self.successful_boundaries + self.boundary_violations
        boundary_success_rate = (
            self.successful_boundaries / total_boundary_checks
            if total_boundary_checks > 0
            else 0
        )

        return {
            "successful_boundaries": self.successful_boundaries,
            "boundary_violations": self.boundary_violations,
            "boundary_success_rate": boundary_success_rate,
            "whitelist_size": len(self.boundary_whitelist_tokens),
            "boundary_checking_enabled": self.enable_boundary_check,
        }

    def get_cache_stats(self):
        """Get combined cache and boundary statistics."""
        cache_stats = self.cache.get_stats()
        boundary_stats = self.get_boundary_stats()
        return {**cache_stats, **boundary_stats}

    def __repr__(self):
        """String representation."""
        stats = self.get_cache_stats()
        repr_str = (
            f"TokenizerWrapper({repr(self.tokenizer)}, "
            f"hit_rate={stats['hit_rate']:.2%}"
        )
        if self.enable_boundary_check:
            repr_str += f", boundary_success={stats['boundary_success_rate']:.2%}"
        repr_str += ")"
        return repr_str

    def apply_chat_template(self, *args, **kwargs):
        tokenize = kwargs.get("tokenize", True)
        if not tokenize:
            return self.tokenizer._original_apply_chat_template(*args, **kwargs)
        else:
            kwargs["tokenize"] = False
            text = self.tokenizer.apply_chat_template(*args, **kwargs)
            prompt_ids = self.encode(text, add_special_tokens=False)
            if self.verify_tokenization_correctness:
                kwargs["tokenize"] = True
                ref_prompt_ids = self.tokenizer._original_apply_chat_template(
                    *args, **kwargs
                )
                assert (
                    ref_prompt_ids == prompt_ids
                ), "Chat template tokenization mismatch"

                print(f"Chat template tokenization passed")

            return prompt_ids
