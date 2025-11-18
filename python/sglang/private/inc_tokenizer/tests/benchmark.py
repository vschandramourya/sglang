#!/usr/bin/env python3
"""
Performance benchmarking for tokenization cache with chat templates.

Benchmarks cache performance using incremental conversation building,
which naturally creates special token boundaries.

python tests/benchmark.py --tokenizer /path/to/model/directory
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer

from sglang.private.inc_tokenizer.cache import TokenizerWrapper
from sglang.private.inc_tokenizer.stats import get_global_stats, reset_global_stats
from sglang.private.inc_tokenizer.tik import create_tiktoken_tokenizer

# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
MODEL_ID = "/scratch/huggingface/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb"
VERIFY_TOKENIZATION_CORRECTNESS = False


def get_tik():
    filename = hf_hub_download(MODEL_ID, "original/tokenizer.model")
    mergeable_ranks = load_tiktoken_bpe(filename)
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    num_reserved_special_tokens = 256
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, num_reserved_special_tokens - 5)
    ]
    num_base_tokens = len(mergeable_ranks)
    special_tokens = {
        token: num_base_tokens + i for i, token in enumerate(special_tokens)
    }
    enc = tiktoken.Encoding(
        name=MODEL_ID,
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    return enc


# from tensorrt_llm._torch.tokenizer.tik import create_tiktoken_tokenizer


def generate_conversation_turn(
    turn_type: str, length: str = "short", prompt_length_mode: str = "stress"
) -> str:
    """Generate a conversation turn with specified characteristics."""

    # Multipliers based on prompt_length_mode
    if prompt_length_mode == "short":
        # Original sizes
        multipliers = {"short": 1, "medium": 5, "long": 20, "very_long": 50}
    elif prompt_length_mode == "medium":
        # Moderate sizes
        multipliers = {"short": 2, "medium": 20, "long": 100, "very_long": 200}
    elif prompt_length_mode == "long":
        # Large sizes
        multipliers = {"short": 5, "medium": 50, "long": 200, "very_long": 500}
    else:  # stress
        # Stress test sizes - with randomization for variety
        multipliers = {
            "short": random.randint(5, 10),
            "medium": random.randint(25, 50),
            "long": random.randint(100, 200),
            "very_long": random.randint(300, 750),
        }

    multiplier = multipliers.get(length, multipliers["medium"])

    if turn_type == "greeting":
        messages = [
            "Hello! How are you today? I was hoping we could discuss some interesting topics about technology and innovation. ",
            "Hi there! I hope you're having a great day. I've been thinking about various aspects of artificial intelligence and would love to explore them with you. ",
            "Greetings! How can I assist you? I'm here to help with any questions about programming, machine learning, or technical topics you might have. ",
        ]
    elif turn_type == "question":
        messages = [
            "Can you explain what machine learning is and how it differs from traditional programming approaches? I'm particularly interested in understanding the fundamental concepts and practical applications. ",
            "What's the difference between AI and machine learning? I've heard these terms used interchangeably but I suspect there are important distinctions that would be helpful to understand. ",
            "How does natural language processing work in modern systems? I'm curious about the underlying mechanisms that allow computers to understand and generate human language. ",
            "Could you help me understand transformers and their role in modern AI? I've been reading about attention mechanisms but the technical details are quite complex. ",
            "What are the main applications of deep learning in industry today? I'm interested in both current use cases and potential future developments in this rapidly evolving field. ",
        ]
    elif turn_type == "explanation":
        messages = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make decisions based on those patterns. The key difference from traditional programming is that instead of writing explicit rules for every scenario, we provide examples and let the system learn the underlying patterns. This approach has revolutionized many fields including computer vision, natural language processing, and recommendation systems. ",
            "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) organized in layers that process information. Each connection has a weight that gets adjusted during training. The network learns by processing training examples and adjusting these weights to minimize prediction errors. Modern deep neural networks can have hundreds of layers and billions of parameters, enabling them to learn incredibly complex patterns in data. ",
            "Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequential data. Unlike recurrent neural networks that process sequences step by step, transformers can process all positions simultaneously, making them much more efficient to train. The attention mechanism allows the model to focus on relevant parts of the input when making predictions. This architecture has become the foundation for most modern language models including GPT, BERT, and their variants. ",
            "Deep learning refers to neural networks with multiple layers that can learn hierarchical representations of data. Each layer learns increasingly abstract features - in image recognition, early layers might detect edges and colors while deeper layers recognize objects and scenes. The 'deep' in deep learning refers to the number of layers in the network. This approach has achieved remarkable success in tasks that were previously thought to require human intelligence, from playing complex games to generating realistic images and text. ",
        ]
    elif turn_type == "code":
        messages = [
            """Here's a comprehensive Python example showing how to implement a basic neural network:
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```
""",
            """You can implement a more complex model using PyTorch with various layers and techniques:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(AdvancedModel, self).__init__()
        self.layers = nn.ModuleList()

        # Build layers dynamically
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.layers.append(nn.BatchNorm1d(dims[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
```
""",
        ]
    else:  # follow-up
        messages = [
            "That makes sense. Can you tell me more about that? I'm particularly interested in understanding the technical details and how this relates to other concepts we've discussed. It would be helpful to see some concrete examples or use cases. ",
            "Interesting! What about the performance implications? I'm wondering about both computational efficiency and accuracy trade-offs. How does this scale with larger datasets or more complex models? ",
            "I see. How does this compare to other approaches? Are there specific scenarios where one method would be preferred over another? What are the key factors to consider when making these architectural decisions? ",
            "Thanks for explaining. Could you provide an example of how this would work in practice? Maybe we could walk through a specific use case or implementation to make the concepts more concrete. ",
        ]

    base_message = random.choice(messages)
    # Add some variation to avoid exact repetition
    if multiplier > 1:
        variations = [
            base_message,
            base_message.replace(".", ". Additionally, "),
            base_message.replace("?", "? Furthermore, "),
            base_message.replace(",", ", moreover,"),
        ]
        parts = []
        for i in range(multiplier):
            parts.append(random.choice(variations))
        return " ".join(parts)
    return base_message


def generate_synthetic_conversations(
    num_conversations: int = 10, prompt_length_mode: str = "stress"
) -> List[List[Dict]]:
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
            # if turn_idx % 2 == 0:  # User turn
            if turn_idx == 0:
                turn_type = "greeting" if random.random() < 0.3 else "question"
            else:
                turn_type = "follow-up" if random.random() < 0.5 else "question"

            # Select length based on mode
            if prompt_length_mode == "short":
                length = random.choice(["short", "short", "medium"])
            elif prompt_length_mode == "medium":
                length = random.choice(["short", "medium", "long"])
            else:  # long or stress
                length = random.choice(["medium", "long", "long", "very_long"])

            conversation.append(
                {
                    "role": "user",
                    "content": generate_conversation_turn(
                        turn_type, length, prompt_length_mode
                    ),
                }
            )
            # else:  # Assistant turn
            turn_type = "explanation" if random.random() < 0.7 else "code"

            # Select length based on mode
            if prompt_length_mode == "short":
                length = random.choice(["short", "medium", "medium"])
            elif prompt_length_mode == "medium":
                length = random.choice(["medium", "long", "long"])
            else:  # long or stress
                length = random.choice(["long", "very_long", "very_long", "very_long"])

            conversation.append(
                {
                    "role": "assistant",
                    "content": generate_conversation_turn(
                        turn_type, length, prompt_length_mode
                    ),
                }
            )

            turn_type = "follow-up" if random.random() < 0.5 else "question"

            # Select length based on mode
            if prompt_length_mode == "short":
                length = random.choice(["short", "short", "medium"])
            elif prompt_length_mode == "medium":
                length = random.choice(["short", "medium", "long"])
            else:  # long or stress
                length = random.choice(["medium", "long", "long", "very_long"])

            conversation.append(
                {
                    "role": "user",
                    "content": generate_conversation_turn(
                        turn_type, length, prompt_length_mode
                    ),
                }
            )

        conversations.append(conversation)

    return conversations


def generate_deepseek_style_conversations(
    num_conversations: int = 10, prompt_length_mode: str = "stress"
) -> List[str]:
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
        num_turns = random.randint(2, 5)

        # Build conversation with special tokens
        text_parts = [special_tokens["bos"]]

        for turn_idx in range(num_turns):
            if turn_idx % 2 == 0:  # User turn
                text_parts.append(special_tokens["user"])
                turn_type = "question" if turn_idx == 0 else "follow-up"

                # Select length based on mode
                if prompt_length_mode == "short":
                    length = random.choice(["short", "short", "medium"])
                elif prompt_length_mode == "medium":
                    length = random.choice(["short", "medium", "long"])
                else:  # long or stress
                    length = random.choice(["medium", "long", "long", "very_long"])

                text_parts.append(
                    generate_conversation_turn(turn_type, length, prompt_length_mode)
                )
                text_parts.append(special_tokens["eot"])
            else:  # Assistant turn
                text_parts.append(special_tokens["assistant"])
                turn_type = "explanation"

                # Select length based on mode
                if prompt_length_mode == "short":
                    length = random.choice(["short", "medium", "medium"])
                elif prompt_length_mode == "medium":
                    length = random.choice(["medium", "long", "long"])
                else:  # long or stress
                    length = random.choice(
                        ["long", "very_long", "very_long", "very_long"]
                    )

                text_parts.append(
                    generate_conversation_turn(turn_type, length, prompt_length_mode)
                )
                text_parts.append(special_tokens["eot"])

        text_parts.append(special_tokens["eos"])
        conversations.append("".join(text_parts))

    return conversations


def benchmark_incremental_chat(
    tokenizer,
    conversations: List[List[Dict]],
    use_cache: bool = True,
    verbose: bool = False,
    tokenizer_path: str = None,
) -> Tuple[float, List[int], Dict]:
    """Benchmark incremental conversation tokenization."""

    if use_cache:
        tokenizer = create_tiktoken_tokenizer(tokenizer_path)
        tokenizer = TokenizerWrapper(
            tokenizer, verify_tokenization_correctness=VERIFY_TOKENIZATION_CORRECTNESS
        )

    start = time.perf_counter()
    token_counts = []
    stats = {
        "total_calls": 0,
        "total_chars": 0,
        "incremental_growth": [],
        "individual_timings": [],
        "per_char_timings": [],
    }

    for conv_idx, conversation in enumerate(conversations):
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
                            conversation=partial_conv,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    else:
                        # For unwrapped tokenizer
                        text = tokenizer.apply_chat_template(
                            conversation=partial_conv,
                            tokenize=False,
                            add_generation_prompt=True,
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

            # print(f'tokenizing text[-200:]: {text[-200:]}')

            # Time individual tokenization
            call_start = time.perf_counter()
            tokens = tokenizer.encode(text, add_special_tokens=False)
            # tokens = tokenizer.encode(text)
            call_elapsed = time.perf_counter() - call_start

            stats["total_calls"] += 1
            stats["total_chars"] += len(text)
            stats["individual_timings"].append(call_elapsed)

            # Per-character timing (microseconds per character)
            per_char_time = (call_elapsed * 1e6) / len(text) if len(text) > 0 else 0
            stats["per_char_timings"].append(per_char_time)

            if verbose:
                print(
                    f"  Conv {conv_idx+1} Turn {i}: {len(text):6d} chars, {len(tokens):5d} tokens, "
                    f"{call_elapsed*1000:6.3f}ms, {per_char_time:.2f}μs/char"
                )

            if i > 1:
                stats["incremental_growth"].append(len(text))

        token_counts.append(len(tokens))

    elapsed = time.perf_counter() - start
    return elapsed, token_counts, stats


def benchmark_deepseek_style(
    tokenizer,
    conversations: List[str],
    use_cache: bool = True,
    verbose: bool = False,
    tokenizer_path: str = None,
) -> Tuple[float, List[int], Dict]:
    """Benchmark with DeepSeek-style special token conversations."""

    if use_cache:
        tokenizer = create_tiktoken_tokenizer(tokenizer_path)
        tokenizer = TokenizerWrapper(
            tokenizer, verify_tokenization_correctness=VERIFY_TOKENIZATION_CORRECTNESS
        )

    start = time.perf_counter()
    token_counts = []
    stats = {
        "total_calls": 0,
        "total_chars": 0,
        "cache_hits": 0,
        "boundary_violations": 0,
        "individual_timings": [],
        "per_char_timings": [],
    }

    for conv_idx, full_conversation in enumerate(conversations):
        # Find special token positions to simulate incremental growth
        special_markers = ["<｜User｜>", "<｜Assistant｜>", "<|EOT|>"]

        # Build incrementally at each special token boundary
        current_pos = 0
        turn = 0
        for marker in special_markers:
            next_pos = full_conversation.find(marker, current_pos + 1)
            if next_pos == -1:
                continue

            # Include the marker itself
            next_pos = next_pos + len(marker)
            partial_text = full_conversation[:next_pos]

            # Time individual tokenization
            call_start = time.perf_counter()
            tokens = tokenizer.encode(partial_text, add_special_tokens=False)
            call_elapsed = time.perf_counter() - call_start

            stats["total_calls"] += 1
            stats["total_chars"] += len(partial_text)
            stats["individual_timings"].append(call_elapsed)

            # Per-character timing (microseconds per character)
            per_char_time = (
                (call_elapsed * 1e6) / len(partial_text) if len(partial_text) > 0 else 0
            )
            stats["per_char_timings"].append(per_char_time)

            if verbose:
                print(
                    f"  Conv {conv_idx+1} Marker {turn+1}: {len(partial_text):6d} chars, {len(tokens):5d} tokens, "
                    f"{call_elapsed*1000:6.3f}ms, {per_char_time:.2f}μs/char"
                )

            current_pos = next_pos
            turn += 1

        # Final full tokenization
        call_start = time.perf_counter()
        tokens = tokenizer.encode(full_conversation, add_special_tokens=False)
        call_elapsed = time.perf_counter() - call_start

        token_counts.append(len(tokens))
        stats["total_calls"] += 1
        stats["total_chars"] += len(full_conversation)
        stats["individual_timings"].append(call_elapsed)

        # Per-character timing
        per_char_time = (
            (call_elapsed * 1e6) / len(full_conversation)
            if len(full_conversation) > 0
            else 0
        )
        stats["per_char_timings"].append(per_char_time)

        if verbose:
            print(
                f"  Conv {conv_idx+1} Final: {len(full_conversation):6d} chars, {len(tokens):5d} tokens, "
                f"{call_elapsed*1000:6.3f}ms, {per_char_time:.2f}μs/char"
            )

    # Get cache stats if using wrapper
    if use_cache and hasattr(tokenizer, "get_cache_stats"):
        cache_stats = tokenizer.get_cache_stats()
        stats["cache_hits"] = cache_stats.get("hits", 0) + cache_stats.get(
            "partial_hits", 0
        )
        stats["boundary_violations"] = cache_stats.get("boundary_violations", 0)

    elapsed = time.perf_counter() - start
    return elapsed, token_counts, stats


def print_timing_statistics(stats_cache: Dict, stats_nocache: Dict, label: str = ""):
    """Print detailed timing statistics."""
    if not stats_cache.get("individual_timings") or not stats_nocache.get(
        "individual_timings"
    ):
        return

    print(f"\n{label} Individual Timing Statistics:")
    print("=" * 60)

    # Without cache statistics
    timings_nocache = (
        np.array(stats_nocache["individual_timings"]) * 1000
    )  # Convert to ms
    per_char_nocache = np.array(stats_nocache["per_char_timings"])

    print("WITHOUT cache:")
    print(
        f"  Timing (ms):  P25={np.percentile(timings_nocache, 25):.3f}, "
        f"P50={np.percentile(timings_nocache, 50):.3f}, "
        f"P75={np.percentile(timings_nocache, 75):.3f}, "
        f"P99={np.percentile(timings_nocache, 99):.3f}"
    )
    print(
        f"  μs/char:      P25={np.percentile(per_char_nocache, 25):.2f}, "
        f"P50={np.percentile(per_char_nocache, 50):.2f}, "
        f"P75={np.percentile(per_char_nocache, 75):.2f}, "
        f"P99={np.percentile(per_char_nocache, 99):.2f}"
    )

    # With cache statistics
    timings_cache = np.array(stats_cache["individual_timings"]) * 1000  # Convert to ms
    per_char_cache = np.array(stats_cache["per_char_timings"])

    print("WITH cache:")
    print(
        f"  Timing (ms):  P25={np.percentile(timings_cache, 25):.3f}, "
        f"P50={np.percentile(timings_cache, 50):.3f}, "
        f"P75={np.percentile(timings_cache, 75):.3f}, "
        f"P99={np.percentile(timings_cache, 99):.3f}"
    )
    print(
        f"  μs/char:      P25={np.percentile(per_char_cache, 25):.2f}, "
        f"P50={np.percentile(per_char_cache, 50):.2f}, "
        f"P75={np.percentile(per_char_cache, 75):.2f}, "
        f"P99={np.percentile(per_char_cache, 99):.2f}"
    )

    # Speedup per percentile
    print("\nSpeedup by percentile:")
    for p in [25, 50, 75, 99]:
        p_nocache = np.percentile(timings_nocache, p)
        p_cache = np.percentile(timings_cache, p)
        speedup = p_nocache / p_cache if p_cache > 0 else 0
        print(f"  P{p:2d}: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tokenization cache with chat templates"
    )
    parser.add_argument("--tokenizer", default=MODEL_ID, help="Tokenizer name or path")
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=6,
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print individual timing for each tokenization",
    )
    parser.add_argument(
        "--prompt-length",
        choices=["short", "medium", "long", "stress"],
        default="stress",
        help="Control prompt lengths (short=original, stress=very long)",
    )
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
        conversations = generate_synthetic_conversations(
            args.num_conversations, args.prompt_length
        )
        print(
            f"Generated {len(conversations)} conversations (mode: {args.prompt_length})"
        )
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
            tokenizer,
            conversations,
            use_cache=False,
            verbose=args.verbose,
            tokenizer_path=args.tokenizer,
        )

        # Reset stats before cached run
        reset_global_stats()

        # Benchmark with cache
        print("Benchmarking WITH cache...")
        time_cache, counts_cache, stats_cache = benchmark_incremental_chat(
            tokenizer,
            conversations,
            use_cache=True,
            verbose=args.verbose,
            tokenizer_path=args.tokenizer,
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

        # Print detailed timing statistics
        if (
            not args.verbose
        ):  # Only print if verbose mode is off (otherwise it's redundant)
            print_timing_statistics(stats_cache, stats_nocache, "Chat Template")

        results["chat"] = {
            "time_nocache": time_nocache,
            "time_cache": time_cache,
            "speedup": time_nocache / time_cache,
            "tokens": sum(counts_nocache),
            "stats_cache": stats_cache,
            "stats_nocache": stats_nocache,
        }

    # Benchmark DeepSeek style
    if args.style in ["deepseek", "both"]:
        print(f"\n{'='*60}")
        print("Benchmarking DeepSeek Style (Special Tokens)")
        print(f"{'='*60}")

        # Generate DeepSeek-style conversations
        deepseek_conversations = generate_deepseek_style_conversations(
            args.num_conversations, args.prompt_length
        )
        print(
            f"Generated {len(deepseek_conversations)} DeepSeek-style conversations (mode: {args.prompt_length})"
        )
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
            tokenizer, deepseek_conversations, use_cache=False, verbose=args.verbose
        )

        # Reset stats before cached run
        reset_global_stats()

        # Benchmark with cache
        print("Benchmarking WITH cache...")
        time_cache, counts_cache, stats_cache = benchmark_deepseek_style(
            tokenizer, deepseek_conversations, use_cache=True, verbose=args.verbose
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

        # Print detailed timing statistics
        if (
            not args.verbose
        ):  # Only print if verbose mode is off (otherwise it's redundant)
            print_timing_statistics(stats_cache, stats_nocache, "DeepSeek Style")

        results["deepseek"] = {
            "time_nocache": time_nocache,
            "time_cache": time_cache,
            "speedup": time_nocache / time_cache,
            "tokens": sum(counts_nocache),
            "cache_hits": stats_cache.get("cache_hits", 0),
            "boundary_violations": stats_cache.get("boundary_violations", 0),
            "stats_cache": stats_cache,
            "stats_nocache": stats_nocache,
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
