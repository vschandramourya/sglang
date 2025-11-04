"""
Tokenization caching module for TGL.

Adapted from TTRT LLM.
https://github.com/togethercomputer/together-TensorRT-LLM/commit/85fa1d15bba6e6dc1d57090ecb51e9eac5193541

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
