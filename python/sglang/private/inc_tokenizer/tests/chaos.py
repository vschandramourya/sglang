#!/usr/bin/env python3
"""
Chaos testing with random chunks to find edge cases.

Tests cache correctness with random split points across many seeds.

python tests/chaos.py --tokenizer /path/to/model/directory --num-seeds 100
python tests/chaos.py --tokenizer /path/to/model/directory --num-seeds 3 --texts-file long_prompts.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from sglang.private.inc_tokenizer.cache import TokenizerWrapper

# import sys
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def load_test_texts(file_path: str = None, count: int = 5) -> List[str]:
    """Load or generate test texts."""
    texts = []

    if file_path and Path(file_path).exists():
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= count:
                    break
                data = json.loads(line) if line.strip() else {}
                texts.append(data.get("text", data.get("prompt", line)))

    if not texts:
        # Generate diverse test cases
        texts = [
            "Simple ASCII text without special characters." * 1000,
            "Unicode: café, naïve, 北京, 🚀, مرحبا" * 1000,
            "Special tokens: <s> </s> <|endoftext|> [CLS] [SEP]" * 1000,
            "Whitespace:\n\ttabs\n  spaces  \r\nCRLF" * 1000,
            "x" * 100000,  # Repetitive text
            "The quick brown fox " * 1000,  # Repeated phrases
        ]

    return texts


def chaos_test_single(
    tokenizer, wrapper, text: str, seed: int
) -> Tuple[bool, List[int]]:
    """Test with random chunks for a single text."""
    random.seed(seed)

    # Generate random chunk boundaries
    num_chunks = random.randint(2, min(20, len(text) // 10))
    boundaries = sorted(
        random.sample(range(1, len(text)), min(num_chunks - 1, len(text) - 1))
    )
    boundaries.append(len(text))

    # Tokenize incrementally with cache
    accumulated = ""
    final_tokens = None
    mismatches = []

    for i, boundary in enumerate(boundaries):
        accumulated = text[:boundary]
        cached_tokens = wrapper.encode(accumulated, add_special_tokens=False)
        direct_tokens = tokenizer.encode(accumulated, add_special_tokens=False)

        if cached_tokens != direct_tokens:
            mismatches.append(i)
            # Find first difference
            for j in range(min(len(cached_tokens), len(direct_tokens))):
                if cached_tokens[j] != direct_tokens[j]:
                    return False, [j, cached_tokens[j], direct_tokens[j]]

        final_tokens = cached_tokens

    return len(mismatches) == 0, final_tokens


def main():
    parser = argparse.ArgumentParser(description="Chaos testing for tokenization cache")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name or path")
    parser.add_argument("--texts-file", help="File with test texts")
    parser.add_argument(
        "--num-texts", type=int, default=5, help="Number of texts to test"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=2, help="Number of random seeds per text"
    )
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed value")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    if "/" in args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, use_fast=True, local_files_only=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # Load test texts
    texts = load_test_texts(args.texts_file, args.num_texts)
    print(f"Testing {len(texts)} texts with {args.num_seeds} seeds each")
    print(f"Total tests: {len(texts) * args.num_seeds}")

    # Run chaos tests
    total_failures = 0
    failure_details = []

    # Overall progress bar for all texts
    with tqdm(total=len(texts), desc="Testing texts", unit="text") as text_pbar:
        for text_idx, text in enumerate(texts):
            wrapper = TokenizerWrapper(tokenizer)  # Fresh cache for each text
            text_preview = text[:50] + "..." if len(text) > 50 else text
            text_pbar.set_description(f"Text {text_idx}: {repr(text_preview)[:30]}")

            failures_for_text = 0

            # Progress bar for seeds within each text
            seed_range = range(args.seed_start, args.seed_start + args.num_seeds)
            for seed in tqdm(
                seed_range, desc=f"  Seeds for text {text_idx}", leave=False
            ):
                success, result = chaos_test_single(tokenizer, wrapper, text, seed)

                if not success:
                    failures_for_text += 1
                    total_failures += 1
                    failure_details.append(
                        {"text_idx": text_idx, "seed": seed, "error": result}
                    )

                # Clear cache periodically to avoid memory issues
                if seed % 20 == 0:
                    wrapper.cache.clear()

            # Report results for this text
            if failures_for_text == 0:
                tqdm.write(f"  ✅ Text {text_idx}: All {args.num_seeds} seeds passed")
            else:
                tqdm.write(
                    f"  ❌ Text {text_idx}: {failures_for_text}/{args.num_seeds} seeds failed"
                )
                # Show first failure for this text
                for detail in failure_details:
                    if detail["text_idx"] == text_idx:
                        tqdm.write(
                            f"     First failure at seed {detail['seed']}: position {detail['error'][0]}"
                        )
                        break

            text_pbar.update(1)

    # Summary
    print("\n" + "=" * 60)
    if total_failures == 0:
        print(f"✅ SUCCESS: All {len(texts) * args.num_seeds} tests passed!")
    else:
        print(
            f"❌ FAILURES: {total_failures}/{len(texts) * args.num_seeds} tests failed"
        )
        print("\nFirst 3 failures:")
        for detail in failure_details[:3]:
            print(
                f"  Text {detail['text_idx']}, Seed {detail['seed']}: {detail['error']}"
            )


if __name__ == "__main__":
    main()
