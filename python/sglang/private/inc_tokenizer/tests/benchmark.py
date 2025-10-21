#!/usr/bin/env python3
"""
Performance benchmarking for tokenization cache with chat templates.

Benchmarks cache performance using incremental conversation building,
which naturally creates special token boundaries.

python tests/benchmark.py --tokenizer /path/to/model/directory
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

from sglang.private.inc_tokenizer.cache import TokenizerWrapper
from sglang.private.inc_tokenizer.stats import get_global_stats, reset_global_stats


def generate_conversation_turn(turn_type: str, length: str = "short") -> str:
    """Generate a conversation turn with specified characteristics."""

    if length == "short":
        multiplier = 1
    elif length == "medium":
        multiplier = 5
    elif length == "long":
        multiplier = 20
    else:
        multiplier = 50  # very long

    if turn_type == "greeting":
        messages = [
            "Hello! How are you today?",
            "Hi there! I hope you're having a great day.",
            "Greetings! How can I assist you?",
        ]
    elif turn_type == "question":
        messages = [
            "Can you explain what machine learning is?",
            "What's the difference between AI and machine learning?",
            "How does natural language processing work?",
            "Could you help me understand transformers?",
            "What are the main applications of deep learning?",
        ]
    elif turn_type == "explanation":
        messages = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. ",
            "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes that process information. ",
            "Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequential data. ",
            "Deep learning refers to neural networks with multiple layers that can learn hierarchical representations of data. ",
        ]
    elif turn_type == "code":
        messages = [
            "Here's a Python example:\n```python\nimport numpy as np\nimport torch\n\ndef example_function(x):\n    return x * 2\n```\n",
            "You can implement this using:\n```python\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layer = torch.nn.Linear(10, 10)\n```\n",
        ]
    else:  # follow-up
        messages = [
            "That makes sense. Can you tell me more about that?",
            "Interesting! What about the performance implications?",
            "I see. How does this compare to other approaches?",
            "Thanks for explaining. Could you provide an example?",
        ]

    base_message = random.choice(messages)
    return base_message * multiplier


def generate_synthetic_conversations(num_conversations: int = 10) -> List[List[Dict]]:
    """Generate synthetic multi-turn conversations."""
    conversations = []

    for i in range(num_conversations):
        # Vary conversation length
        num_turns = random.randint(2, 10)

        conversation = []

        # Always start with a system message
        conversation.append(
            {
                "role": "system",
                "content": "You are a helpful AI assistant with expertise in machine learning and programming.",
            }
        )

        # Generate turns
        for turn_idx in range(num_turns):
            if turn_idx % 2 == 0:  # User turn
                if turn_idx == 0:
                    turn_type = "greeting" if random.random() < 0.3 else "question"
                else:
                    turn_type = "follow-up" if random.random() < 0.5 else "question"

                length = random.choice(["short", "medium", "long"])
                conversation.append(
                    {
                        "role": "user",
                        "content": generate_conversation_turn(turn_type, length),
                    }
                )
            else:  # Assistant turn
                turn_type = "explanation" if random.random() < 0.7 else "code"
                length = random.choice(["medium", "long", "very_long"])
                conversation.append(
                    {
                        "role": "assistant",
                        "content": generate_conversation_turn(turn_type, length),
                    }
                )

        conversations.append(conversation)

    return conversations


def generate_deepseek_style_conversations(num_conversations: int = 10) -> List[str]:
    """Generate conversations using DeepSeek-style special tokens."""
    conversations = []

    special_tokens = {
        "bos": "<｜begin▁of▁sentence｜>",
        "eos": "<｜end▁of▁sentence｜>",
        "user": "<｜User｜>",
        "assistant": "<｜Assistant｜>",
        "eot": "<|EOT|>",
    }

    for i in range(num_conversations):
        num_turns = random.randint(2, 8)

        # Build conversation with special tokens
        text_parts = [special_tokens["bos"]]

        for turn_idx in range(num_turns):
            if turn_idx % 2 == 0:  # User turn
                text_parts.append(special_tokens["user"])
                turn_type = "question" if turn_idx == 0 else "follow-up"
                length = random.choice(["short", "medium", "long"])
                text_parts.append(generate_conversation_turn(turn_type, length))
                text_parts.append(special_tokens["eot"])
            else:  # Assistant turn
                text_parts.append(special_tokens["assistant"])
                turn_type = "explanation"
                length = random.choice(["medium", "long", "very_long"])
                text_parts.append(generate_conversation_turn(turn_type, length))
                text_parts.append(special_tokens["eot"])

        text_parts.append(special_tokens["eos"])
        conversations.append("".join(text_parts))

    return conversations


def benchmark_incremental_chat(
    tokenizer, conversations: List[List[Dict]], use_cache: bool = True
) -> Tuple[float, List[int], Dict]:
    """Benchmark incremental conversation tokenization."""

    if use_cache:
        tokenizer = TokenizerWrapper(tokenizer)

    start = time.perf_counter()
    token_counts = []
    stats = {"total_calls": 0, "total_chars": 0, "incremental_growth": []}

    for conversation in conversations:
        # Build conversation incrementally (simulating real chat)
        for i in range(1, len(conversation) + 1):
            partial_conv = conversation[:i]

            # Apply chat template if available
            if hasattr(tokenizer, "apply_chat_template") or (
                hasattr(tokenizer, "tokenizer")
                and hasattr(tokenizer.tokenizer, "apply_chat_template")
            ):
                try:
                    if hasattr(tokenizer, "tokenizer"):
                        # For wrapped tokenizer
                        text = tokenizer.tokenizer.apply_chat_template(
                            partial_conv,
                            tokenize=False,
                            add_generation_prompt=(i == len(conversation)),
                        )
                    else:
                        # For unwrapped tokenizer
                        text = tokenizer.apply_chat_template(
                            partial_conv,
                            tokenize=False,
                            add_generation_prompt=(i == len(conversation)),
                        )
                except:
                    # Fallback to simple format
                    text = "\n".join(
                        [f"{msg['role']}: {msg['content']}" for msg in partial_conv]
                    )
            else:
                # Simple fallback format
                text = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in partial_conv]
                )

            tokens = tokenizer.encode(text, add_special_tokens=False)
            stats["total_calls"] += 1
            stats["total_chars"] += len(text)

            if i > 1:
                stats["incremental_growth"].append(len(text))

        token_counts.append(len(tokens))

    elapsed = time.perf_counter() - start
    return elapsed, token_counts, stats


def benchmark_deepseek_style(
    tokenizer, conversations: List[str], use_cache: bool = True
) -> Tuple[float, List[int], Dict]:
    """Benchmark with DeepSeek-style special token conversations."""

    if use_cache:
        tokenizer = TokenizerWrapper(tokenizer)

    start = time.perf_counter()
    token_counts = []
    stats = {
        "total_calls": 0,
        "total_chars": 0,
        "cache_hits": 0,
        "boundary_violations": 0,
    }

    for full_conversation in conversations:
        # Find special token positions to simulate incremental growth
        special_markers = ["<｜User｜>", "<｜Assistant｜>", "<|EOT|>"]

        # Build incrementally at each special token boundary
        current_pos = 0
        for marker in special_markers:
            next_pos = full_conversation.find(marker, current_pos + 1)
            if next_pos == -1:
                continue

            # Include the marker itself
            next_pos = next_pos + len(marker)
            partial_text = full_conversation[:next_pos]

            tokens = tokenizer.encode(partial_text, add_special_tokens=False)
            stats["total_calls"] += 1
            stats["total_chars"] += len(partial_text)

            current_pos = next_pos

        # Final full tokenization
        tokens = tokenizer.encode(full_conversation, add_special_tokens=False)
        token_counts.append(len(tokens))
        stats["total_calls"] += 1
        stats["total_chars"] += len(full_conversation)

    # Get cache stats if using wrapper
    if use_cache and hasattr(tokenizer, "get_cache_stats"):
        cache_stats = tokenizer.get_cache_stats()
        stats["cache_hits"] = cache_stats.get("hits", 0) + cache_stats.get(
            "partial_hits", 0
        )
        stats["boundary_violations"] = cache_stats.get("boundary_violations", 0)

    elapsed = time.perf_counter() - start
    return elapsed, token_counts, stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tokenization cache with chat templates"
    )
    parser.add_argument(
        "--tokenizer", default="meta-llama/Llama-2-7b-hf", help="Tokenizer name or path"
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=10,
        help="Number of conversations to generate",
    )
    parser.add_argument(
        "--style",
        choices=["chat", "deepseek", "both"],
        default="chat",
        help="Conversation style to benchmark",
    )
    parser.add_argument(
        "--show-stats", action="store_true", help="Show detailed statistics"
    )
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    if Path(args.tokenizer).exists():
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, use_fast=True, local_files_only=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    print(f"Tokenizer type: {tokenizer.__class__.__name__}")
    print(f"Has chat template: {hasattr(tokenizer, 'apply_chat_template')}")

    results = {}

    # Benchmark chat template style
    if args.style in ["chat", "both"]:
        print(f"\n{'='*60}")
        print("Benchmarking Chat Template Style Conversations")
        print(f"{'='*60}")

        # Generate conversations
        conversations = generate_synthetic_conversations(args.num_conversations)
        print(f"Generated {len(conversations)} conversations")
        print(f"Conversation lengths: {[len(c) for c in conversations[:5]]}... turns")

        # Warmup
        if args.warmup > 0:
            print(f"Running {args.warmup} warmup iterations...")
            for _ in range(args.warmup):
                benchmark_incremental_chat(
                    tokenizer, conversations[:2], use_cache=False
                )

        # Benchmark without cache
        print("\nBenchmarking WITHOUT cache...")
        time_nocache, counts_nocache, stats_nocache = benchmark_incremental_chat(
            tokenizer, conversations, use_cache=False
        )

        # Reset stats before cached run
        reset_global_stats()

        # Benchmark with cache
        print("Benchmarking WITH cache...")
        time_cache, counts_cache, stats_cache = benchmark_incremental_chat(
            tokenizer, conversations, use_cache=True
        )

        # Verify correctness
        if counts_nocache != counts_cache:
            print("❌ ERROR: Token counts don't match!")
            for i, (nc, c) in enumerate(zip(counts_nocache, counts_cache)):
                if nc != c:
                    print(f"  Conversation {i}: {nc} vs {c} tokens")
        else:
            print("✅ Correctness verified")

        # Report results
        print(f"\nChat Template Results:")
        print(f"  Without cache: {time_nocache:.3f}s")
        print(f"  With cache:    {time_cache:.3f}s")
        print(f"  Speedup:       {time_nocache/time_cache:.2f}x")
        print(f"  Total tokens:  {sum(counts_nocache):,}")
        print(f"  Total calls:   {stats_nocache['total_calls']}")
        print(f"  Total chars:   {stats_nocache['total_chars']:,}")

        results["chat"] = {
            "time_nocache": time_nocache,
            "time_cache": time_cache,
            "speedup": time_nocache / time_cache,
            "tokens": sum(counts_nocache),
        }

    # Benchmark DeepSeek style
    if args.style in ["deepseek", "both"]:
        print(f"\n{'='*60}")
        print("Benchmarking DeepSeek Style (Special Tokens)")
        print(f"{'='*60}")

        # Generate DeepSeek-style conversations
        deepseek_conversations = generate_deepseek_style_conversations(
            args.num_conversations
        )
        print(f"Generated {len(deepseek_conversations)} DeepSeek-style conversations")
        print(f"Sample conversation length: {len(deepseek_conversations[0])} chars")

        # Show sample special tokens
        sample = deepseek_conversations[0][:200]
        print(f"Sample start: {repr(sample)}...")

        # Warmup
        if args.warmup > 0:
            print(f"Running {args.warmup} warmup iterations...")
            for _ in range(args.warmup):
                benchmark_deepseek_style(
                    tokenizer, deepseek_conversations[:2], use_cache=False
                )

        # Benchmark without cache
        print("\nBenchmarking WITHOUT cache...")
        time_nocache, counts_nocache, stats_nocache = benchmark_deepseek_style(
            tokenizer, deepseek_conversations, use_cache=False
        )

        # Reset stats before cached run
        reset_global_stats()

        # Benchmark with cache
        print("Benchmarking WITH cache...")
        time_cache, counts_cache, stats_cache = benchmark_deepseek_style(
            tokenizer, deepseek_conversations, use_cache=True
        )

        # Verify correctness
        if counts_nocache != counts_cache:
            print("❌ ERROR: Token counts don't match!")
            for i, (nc, c) in enumerate(zip(counts_nocache, counts_cache)):
                if nc != c:
                    print(f"  Conversation {i}: {nc} vs {c} tokens")
        else:
            print("✅ Correctness verified")

        # Report results
        print(f"\nDeepSeek Style Results:")
        print(f"  Without cache: {time_nocache:.3f}s")
        print(f"  With cache:    {time_cache:.3f}s")
        print(f"  Speedup:       {time_nocache/time_cache:.2f}x")
        print(f"  Total tokens:  {sum(counts_nocache):,}")
        print(f"  Total calls:   {stats_nocache['total_calls']}")
        print(f"  Cache hits:    {stats_cache.get('cache_hits', 0)}")
        print(f"  Boundary violations: {stats_cache.get('boundary_violations', 0)}")

        results["deepseek"] = {
            "time_nocache": time_nocache,
            "time_cache": time_cache,
            "speedup": time_nocache / time_cache,
            "tokens": sum(counts_nocache),
            "cache_hits": stats_cache.get("cache_hits", 0),
            "boundary_violations": stats_cache.get("boundary_violations", 0),
        }

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Summary Comparison")
        print(f"{'='*60}")
        for style, res in results.items():
            print(
                f"{style.capitalize():10s}: {res['speedup']:.2f}x speedup, {res['tokens']:,} tokens"
            )

    # Show detailed statistics if requested
    if args.show_stats:
        stats = get_global_stats()
        if stats:
            print("\n" + "=" * 60)
            print("Detailed Cache Statistics")
            print("=" * 60)
            print(stats.detailed_report())


if __name__ == "__main__":
    main()
