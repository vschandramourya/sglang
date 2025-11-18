#!/usr/bin/env python3
"""
Statistics and timing tracking for tokenization cache.

Designed for minimal overhead with detailed performance insights.
"""

import statistics
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List


def milliseconds(seconds: float) -> float:
    """Convert seconds to milliseconds."""
    return seconds * 1_000


@dataclass
class CacheTimestamps:
    """
    Timestamps for a single cache operation.

    Uses perf_counter for high precision timing.
    """

    # Request received
    start: float

    # Cache lookup completed
    cache_lookup: float = 0.0

    # Prefix search completed (if performed)
    prefix_search: float = 0.0

    # Stability computation completed (if performed)
    stability_computed: float = 0.0

    # Tokenization completed (if performed)
    tokenization_done: float = 0.0

    # Cache store completed
    cache_stored: float = 0.0

    # Final response time
    completed: float = 0.0

    # Operation metadata
    text_length: int = 0
    token_count: int = 0
    cache_hit: bool = False
    prefix_hit: bool = False
    prefix_reused_tokens: int = 0
    prefix_tokenized_chars: int = 0

    @classmethod
    def initialize(cls) -> "CacheTimestamps":
        """Start a new timing session."""
        return cls(start=perf_counter())

    def mark_cache_lookup(self) -> "CacheTimestamps":
        """Mark cache lookup completion."""
        self.cache_lookup = perf_counter()
        return self

    def mark_prefix_search(self) -> "CacheTimestamps":
        """Mark prefix search completion."""
        self.prefix_search = perf_counter()
        return self

    def mark_stability(self) -> "CacheTimestamps":
        """Mark stability computation completion."""
        self.stability_computed = perf_counter()
        return self

    def mark_tokenization(self) -> "CacheTimestamps":
        """Mark tokenization completion."""
        self.tokenization_done = perf_counter()
        return self

    def mark_cache_store(self) -> "CacheTimestamps":
        """Mark cache storage completion."""
        self.cache_stored = perf_counter()
        return self

    def finalize(self) -> "CacheTimestamps":
        """Mark operation completion."""
        self.completed = perf_counter()
        return self

    def derived_metrics(self) -> Dict[str, float]:
        """Calculate derived timing metrics."""
        result = {
            "total_time_ms": milliseconds(self.completed - self.start),
            "cache_lookup_ms": (
                milliseconds(self.cache_lookup - self.start) if self.cache_lookup else 0
            ),
        }

        if self.prefix_search:
            result["prefix_search_ms"] = milliseconds(
                self.prefix_search - self.cache_lookup
            )

        if self.stability_computed:
            prev = self.prefix_search if self.prefix_search else self.cache_lookup
            result["stability_compute_ms"] = milliseconds(
                self.stability_computed - prev
            )

        if self.tokenization_done:
            prev = (
                self.stability_computed
                if self.stability_computed
                else (self.prefix_search if self.prefix_search else self.cache_lookup)
            )
            result["tokenization_ms"] = milliseconds(self.tokenization_done - prev)

            # Calculate tokenization throughput
            if self.text_length > 0:
                tokenization_time = self.tokenization_done - prev
                result["chars_per_second"] = (
                    self.text_length / tokenization_time if tokenization_time > 0 else 0
                )
                result["tokens_per_second"] = (
                    self.token_count / tokenization_time if tokenization_time > 0 else 0
                )

        if self.cache_stored:
            prev = self.tokenization_done if self.tokenization_done else self.completed
            result["cache_store_ms"] = milliseconds(self.cache_stored - prev)

        # Add efficiency metrics
        if self.prefix_hit and self.text_length > 0:
            result["prefix_reuse_ratio"] = self.prefix_reused_tokens / max(
                1, self.token_count
            )
            result["chars_tokenized_ratio"] = (
                self.prefix_tokenized_chars / self.text_length
            )

        return result


@dataclass
class CacheStatistics:
    """
    Aggregated statistics for cache operations.

    Thread-safe accumulator for performance metrics.
    """

    # Counters
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    prefix_hits: int = 0

    # Text statistics
    total_chars_processed: int = 0
    total_tokens_generated: int = 0
    total_prefix_tokens_reused: int = 0
    total_chars_tokenized: int = 0  # Chars actually tokenized (not from cache)

    # Timing aggregates (all in milliseconds)
    timings: List[CacheTimestamps] = field(default_factory=list)

    # Detailed timing buckets (in milliseconds)
    cache_lookup_times: List[float] = field(default_factory=list)
    prefix_search_times: List[float] = field(default_factory=list)
    stability_compute_times: List[float] = field(default_factory=list)
    tokenization_times: List[float] = field(default_factory=list)
    cache_store_times: List[float] = field(default_factory=list)
    total_times: List[float] = field(default_factory=list)

    def record(self, timestamp: CacheTimestamps) -> None:
        """Record a completed operation."""
        self.total_requests += 1

        if timestamp.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if timestamp.prefix_hit:
            self.prefix_hits += 1
            self.total_prefix_tokens_reused += timestamp.prefix_reused_tokens
            self.total_chars_tokenized += timestamp.prefix_tokenized_chars
        elif not timestamp.cache_hit:
            # Full tokenization
            self.total_chars_tokenized += timestamp.text_length

        self.total_chars_processed += timestamp.text_length
        self.total_tokens_generated += timestamp.token_count

        # Store timing data
        self.timings.append(timestamp)
        metrics = timestamp.derived_metrics()

        if "cache_lookup_ms" in metrics:
            self.cache_lookup_times.append(metrics["cache_lookup_ms"])
        if "prefix_search_ms" in metrics:
            self.prefix_search_times.append(metrics["prefix_search_ms"])
        if "stability_compute_ms" in metrics:
            self.stability_compute_times.append(metrics["stability_compute_ms"])
        if "tokenization_ms" in metrics:
            self.tokenization_times.append(metrics["tokenization_ms"])
        if "cache_store_ms" in metrics:
            self.cache_store_times.append(metrics["cache_store_ms"])
        if "total_time_ms" in metrics:
            self.total_times.append(metrics["total_time_ms"])

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.total_requests:
            return {"message": "No requests recorded"}

        result = {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "prefix_hits": self.prefix_hits,
            "cache_hit_rate": self.cache_hits / self.total_requests,
            "prefix_hit_rate": self.prefix_hits / max(1, self.cache_misses),
            "total_chars": self.total_chars_processed,
            "total_tokens": self.total_tokens_generated,
            "avg_tokens_per_request": self.total_tokens_generated / self.total_requests,
        }

        # Efficiency metrics
        if self.total_chars_processed > 0:
            result["tokenization_saved_ratio"] = 1 - (
                self.total_chars_tokenized / self.total_chars_processed
            )

        if self.prefix_hits > 0:
            result["avg_prefix_tokens_reused"] = (
                self.total_prefix_tokens_reused / self.prefix_hits
            )

        # Timing statistics
        def add_timing_stats(name: str, times: List[float]) -> None:
            if times:
                result[f"{name}_avg_ms"] = statistics.mean(times)
                result[f"{name}_median_ms"] = statistics.median(times)
                if len(times) > 1:
                    result[f"{name}_stdev_ms"] = statistics.stdev(times)
                result[f"{name}_min_ms"] = min(times)
                result[f"{name}_max_ms"] = max(times)

                # Add percentiles for total time
                if name == "total" and len(times) >= 10:
                    sorted_times = sorted(times)
                    result["total_p50_ms"] = sorted_times[len(sorted_times) // 2]
                    result["total_p95_ms"] = sorted_times[int(len(sorted_times) * 0.95)]
                    result["total_p99_ms"] = sorted_times[int(len(sorted_times) * 0.99)]

        add_timing_stats("cache_lookup", self.cache_lookup_times)
        add_timing_stats("prefix_search", self.prefix_search_times)
        add_timing_stats("stability", self.stability_compute_times)
        add_timing_stats("tokenization", self.tokenization_times)
        add_timing_stats("cache_store", self.cache_store_times)
        add_timing_stats("total", self.total_times)

        return result

    def detailed_report(self) -> str:
        """Generate a detailed text report."""
        summary = self.summary()

        if "message" in summary:
            return summary["message"]

        lines = [
            "=== TOKENIZATION CACHE STATISTICS ===",
            "",
            "OVERALL PERFORMANCE:",
            f"  Total requests: {summary['total_requests']:,}",
            f"  Cache hits: {summary['cache_hits']:,} ({summary['cache_hit_rate']:.1%})",
            f"  Cache misses: {summary['cache_misses']:,}",
            f"  Prefix hits: {summary['prefix_hits']:,} ({summary['prefix_hit_rate']:.1%} of misses)",
            "",
            "DATA PROCESSED:",
            f"  Total characters: {summary['total_chars']:,}",
            f"  Total tokens: {summary['total_tokens']:,}",
            f"  Avg tokens/request: {summary['avg_tokens_per_request']:.1f}",
        ]

        if "tokenization_saved_ratio" in summary:
            lines.append(
                f"  Tokenization saved: {summary['tokenization_saved_ratio']:.1%}"
            )

        if "avg_prefix_tokens_reused" in summary:
            lines.append(
                f"  Avg prefix tokens reused: {summary['avg_prefix_tokens_reused']:.1f}"
            )

        lines.extend(["", "TIMING BREAKDOWN (milliseconds):"])

        # Format timing table
        timing_categories = [
            ("Cache Lookup", "cache_lookup"),
            ("Prefix Search", "prefix_search"),
            ("Stability Compute", "stability"),
            ("Tokenization", "tokenization"),
            ("Cache Store", "cache_store"),
            ("Total", "total"),
        ]

        lines.append(
            f"  {'Operation':<20} {'Avg':>10} {'Median':>10} {'Min':>10} {'Max':>10}"
        )
        lines.append("  " + "-" * 60)

        for display_name, key in timing_categories:
            avg_key = f"{key}_avg_ms"
            if avg_key in summary:
                avg_val = summary[avg_key]
                median_val = summary.get(f"{key}_median_ms", 0)
                min_val = summary.get(f"{key}_min_ms", 0)
                max_val = summary.get(f"{key}_max_ms", 0)
                lines.append(
                    f"  {display_name:<20} {avg_val:>10.3f} {median_val:>10.3f} "
                    f"{min_val:>10.3f} {max_val:>10.3f}"
                )

        if "total_p50_ms" in summary:
            lines.extend(
                [
                    "",
                    "LATENCY PERCENTILES (milliseconds):",
                    f"  P50: {summary['total_p50_ms']:.3f}",
                    f"  P95: {summary['total_p95_ms']:.3f}",
                    f"  P99: {summary['total_p99_ms']:.3f}",
                ]
            )

        # Throughput if available
        if self.tokenization_times:
            total_tokenization_s = (
                sum(self.tokenization_times) / 1_000
            )  # Convert ms to seconds
            if total_tokenization_s > 0:
                throughput_tokens = self.total_chars_tokenized / 4  # Rough estimate
                lines.extend(
                    [
                        "",
                        "THROUGHPUT:",
                        f"  Tokenization rate: {throughput_tokens / total_tokenization_s:.0f} tokens/sec",
                        f"  Character rate: {self.total_chars_tokenized / total_tokenization_s:.0f} chars/sec",
                    ]
                )

        return "\n".join(lines)


# Global statistics instance for easy access
_global_stats = CacheStatistics()


def get_global_stats() -> CacheStatistics:
    """Get the global statistics instance."""
    return _global_stats


def reset_global_stats() -> None:
    """Reset global statistics."""
    global _global_stats
    _global_stats = CacheStatistics()
