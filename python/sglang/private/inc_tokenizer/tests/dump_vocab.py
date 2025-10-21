#!/usr/bin/env python3
"""
Dump tokenizer vocabulary to a file for inspection.

Usage:
    python tests/dump_vocab.py --tokenizer meta-llama/Llama-2-7b-hf
    python tests/dump_vocab.py --tokenizer /path/to/model --output vocab.txt
    python tests/dump_vocab.py --tokenizer gpt2 --format json
    python tests/dump_vocab.py --tokenizer gpt2 --filter "Hello"
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def load_tokenizer(tokenizer_path):
    """Load tokenizer from path or name."""
    try:
        if Path(tokenizer_path).exists():
            return AutoTokenizer.from_pretrained(
                tokenizer_path, use_fast=True, local_files_only=True
            )
        else:
            return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit(1)


def dump_vocab(
    tokenizer, output_file, format_type="tsv", filter_text=None, show_special=True
):
    """Dump vocabulary to file."""

    # Get vocabulary
    vocab = tokenizer.get_vocab()

    # Get special tokens info
    special_tokens = set()
    if hasattr(tokenizer, "all_special_tokens"):
        special_tokens = set(tokenizer.all_special_ids)

    # Sort by token ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    # Filter if requested
    if filter_text:
        sorted_vocab = [
            (k, v) for k, v in sorted_vocab if filter_text.lower() in k.lower()
        ]
        print(f"Filtered to {len(sorted_vocab)} tokens containing '{filter_text}'")

    print(f"Writing {len(sorted_vocab)} tokens to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        if format_type == "json":
            # JSON format
            output = {}
            for token_str, token_id in sorted_vocab:
                entry = {
                    "id": token_id,
                    "token": token_str,
                    "bytes": list(token_str.encode("utf-8", errors="ignore")),
                    "special": token_id in special_tokens,
                }
                output[token_id] = entry
            json.dump(output, f, indent=2, ensure_ascii=False)

        elif format_type == "readable":
            # Human-readable format with details
            f.write(
                "Token ID | Special | Length | Token String | Hex Bytes | Decoded\n"
            )
            f.write("-" * 80 + "\n")

            for token_str, token_id in sorted_vocab:
                is_special = token_id in special_tokens
                if not show_special and is_special:
                    continue

                # Try to decode the token alone
                try:
                    decoded = tokenizer.decode([token_id])
                except:
                    decoded = "[DECODE ERROR]"

                # Get byte representation
                token_bytes = token_str.encode("utf-8", errors="ignore")
                hex_bytes = " ".join(f"{b:02x}" for b in token_bytes)

                special_marker = "*" if is_special else " "
                f.write(
                    f"{token_id:7d} | {special_marker:7s} | {len(token_bytes):6d} | {repr(token_str):30s} | {hex_bytes:20s} | {repr(decoded)}\n"
                )

        else:  # Default TSV format
            # Simple TSV format
            f.write("token_id\ttoken_string\tis_special\tdecoded\n")

            for token_str, token_id in sorted_vocab:
                is_special = token_id in special_tokens
                if not show_special and is_special:
                    continue

                # Try to decode the token alone
                try:
                    decoded = tokenizer.decode([token_id])
                    # Escape tabs and newlines for TSV
                    decoded = (
                        decoded.replace("\t", "\\t")
                        .replace("\n", "\\n")
                        .replace("\r", "\\r")
                    )
                except:
                    decoded = "[DECODE ERROR]"

                # Escape the token string for TSV
                token_escaped = (
                    token_str.replace("\t", "\\t")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                )

                f.write(f"{token_id}\t{token_escaped}\t{is_special}\t{decoded}\n")

    print(f"Vocabulary dumped to {output_file}")

    # Print summary statistics
    print("\nVocabulary Statistics:")
    print(f"  Total tokens: {len(vocab):,}")
    print(f"  Special tokens: {len(special_tokens):,}")
    print(f"  Regular tokens: {len(vocab) - len(special_tokens):,}")

    if hasattr(tokenizer, "vocab_size"):
        print(f"  Vocab size (reported): {tokenizer.vocab_size:,}")

    # Find interesting patterns
    if not filter_text:
        # Count tokens by prefix
        prefixes = {}
        for token_str, _ in sorted_vocab[:1000]:  # Check first 1000
            if token_str.startswith("▁"):  # Common sentencepiece prefix
                prefixes["sentencepiece"] = prefixes.get("sentencepiece", 0) + 1
            elif token_str.startswith("Ġ"):  # GPT2 style space
                prefixes["gpt2_space"] = prefixes.get("gpt2_space", 0) + 1
            elif token_str.startswith("##"):  # BERT style wordpiece
                prefixes["wordpiece"] = prefixes.get("wordpiece", 0) + 1

        if prefixes:
            print("\nToken style detected (from first 1000 tokens):")
            for style, count in prefixes.items():
                print(f"  {style}: {count}")

    # Show example tokens
    print("\nExample tokens:")
    examples = (
        sorted_vocab[:5] + sorted_vocab[-5:] if len(sorted_vocab) > 10 else sorted_vocab
    )
    for token_str, token_id in examples[:10]:
        try:
            decoded = tokenizer.decode([token_id])
            print(f"  {token_id:6d}: {repr(token_str):20s} -> {repr(decoded)}")
        except:
            print(f"  {token_id:6d}: {repr(token_str):20s} -> [DECODE ERROR]")


def main():
    parser = argparse.ArgumentParser(
        description="Dump tokenizer vocabulary to file for inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dump vocabulary as TSV (default)
  python tests/dump_vocab.py --tokenizer meta-llama/Llama-2-7b-hf

  # Output to specific file
  python tests/dump_vocab.py --tokenizer gpt2 --output gpt2_vocab.txt

  # JSON format with full details
  python tests/dump_vocab.py --tokenizer gpt2 --format json --output vocab.json

  # Human-readable format with byte info
  python tests/dump_vocab.py --tokenizer gpt2 --format readable

  # Filter tokens containing specific text
  python tests/dump_vocab.py --tokenizer gpt2 --filter "hello"

  # Exclude special tokens
  python tests/dump_vocab.py --tokenizer gpt2 --no-special
        """,
    )

    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Path to tokenizer or HuggingFace model name",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: [tokenizer_name]_vocab.[ext])",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["tsv", "json", "readable"],
        default="tsv",
        help="Output format (tsv, json, or readable)",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter tokens containing this text (case-insensitive)",
    )

    parser.add_argument(
        "--no-special", action="store_true", help="Exclude special tokens from output"
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Loaded: {tokenizer.__class__.__name__}")

    # Determine output file
    if args.output is None:
        # Create default filename based on tokenizer name
        tokenizer_name = args.tokenizer.replace("/", "_").replace("\\", "_")
        if len(tokenizer_name) > 50:
            tokenizer_name = tokenizer_name[-50:]  # Truncate if too long

        ext = "json" if args.format == "json" else "txt"
        args.output = f"{tokenizer_name}_vocab.{ext}"

    # Dump vocabulary
    dump_vocab(
        tokenizer,
        args.output,
        format_type=args.format,
        filter_text=args.filter,
        show_special=not args.no_special,
    )


if __name__ == "__main__":
    main()
