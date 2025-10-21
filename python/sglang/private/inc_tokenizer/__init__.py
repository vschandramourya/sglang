"""
Tokenization caching module for TGL.

Adapted from TTRT LLM.
https://github.com/togethercomputer/together-TensorRT-LLM/commit/e9de23667dfce64da8b949d4a4cdada1e57f4c5c

Provides caching and optimization for tokenizer operations with boundary-safe
prefix matching for incremental text tokenization.
"""

from .cache import TokenizationCache, TokenizerWrapper
from .stats import CacheStatistics, get_global_stats, reset_global_stats

__all__ = [
    "TokenizerWrapper",
    "TokenizationCache",
    "CacheStatistics",
    "get_global_stats",
    "reset_global_stats",
]
