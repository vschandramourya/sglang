from collections import Counter
from typing import List, Optional


def compare_tokenizations(
    cached_tokens: List[int],
    reference_tokens: List[int],
    cached_time: float,
    reference_time: float,
    prompt_length: Optional[int] = None,
) -> None:
    """
    Compare cached vs non-cached tokenization results using frequency counting.

    This uses frequency counting to detect actual token differences, not just shifts.
    For example, if one token is inserted early, the rest of the sequence may be
    identical just shifted by one position. Frequency counting reveals the true
    difference is just one token.

    Args:
        cached_tokens: Token IDs from cached tokenization
        reference_tokens: Token IDs from non-cached (reference) tokenization
        cached_time: Time taken for cached tokenization (seconds)
        reference_time: Time taken for non-cached tokenization (seconds)
        prompt_length: Optional length of the prompt in characters
    """
    # Log timing information
    print(f"cached tokenize {len(cached_tokens)} token prompt time: {cached_time}")
    print(
        f"non-cached tokenize {len(reference_tokens)} token prompt time: {reference_time}"
    )

    # Use frequency counting to find actual differences
    cached_freq = Counter(cached_tokens)
    reference_freq = Counter(reference_tokens)

    # Find tokens that differ in frequency
    all_tokens = set(cached_freq.keys()) | set(reference_freq.keys())
    freq_diffs = {}

    for token in all_tokens:
        cached_count = cached_freq.get(token, 0)
        ref_count = reference_freq.get(token, 0)
        if cached_count != ref_count:
            freq_diffs[token] = abs(cached_count - ref_count)

    # Calculate the actual number of token differences
    total_diff_tokens = sum(freq_diffs.values())
    # The real difference is half of the total (since each mismatch is counted twice -
    # once as missing and once as extra)
    actual_differences = total_diff_tokens // 2
    if total_diff_tokens % 2 != 0:
        actual_differences += 1  # Handle odd case (shouldn't happen normally)

    # Also check if sequences are exactly equal (same order)
    sequences_match = cached_tokens == reference_tokens

    # Calculate statistics
    total_tokens = max(len(cached_tokens), len(reference_tokens))

    if sequences_match:
        print(
            f"cached tokenize {len(cached_tokens)} token prompt matches non-cached tokenize {len(reference_tokens)} token prompt"
        )
    else:
        # Use frequency-based difference for the percentage
        if actual_differences == 0:
            # Same tokens but different order (rare but possible)
            print(
                f"cached tokenize {len(cached_tokens)} token prompt matches non-cached tokenize {len(reference_tokens)} token prompt (same tokens, different order)"
            )
        else:
            mismatch_rate = (
                (actual_differences / total_tokens) * 100 if total_tokens > 0 else 0
            )

            # Build detailed message
            if len(cached_tokens) != len(reference_tokens):
                mismatch_msg = (
                    f"{actual_differences} / {total_tokens} tokens differ ({mismatch_rate:.4f}%) - "
                    f"length: cached={len(cached_tokens)}, ref={len(reference_tokens)}"
                )
            else:
                mismatch_msg = f"{actual_differences} / {total_tokens} tokens differ ({mismatch_rate:.4f}%)"

            # Add details about specific token differences for small mismatches
            if actual_differences <= 5 and len(freq_diffs) > 0:
                diff_details = []
                for token, count in list(freq_diffs.items())[:5]:
                    diff_details.append(f"token_{token}:{count}")
                mismatch_msg += f' [{", ".join(diff_details)}]'

            print(
                f"cached tokenize {len(cached_tokens)} token prompt does not match non-cached tokenize {len(reference_tokens)} token prompt - {mismatch_msg}"
            )

    # Log speedup
    if cached_time > 0 and reference_time > 0:
        speedup = reference_time / cached_time
        print(f"speedup: {speedup:.2f}x")
