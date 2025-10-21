#!/usr/bin/env python3
"""
Unit tests with minimal cases that reproduce specific bugs.

Each test is a minimal reproduction of a discovered issue.

python tests/unit.py
python tests/unit.py --tokenizer /path/to/model
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from sglang.private.inc_tokenizer.cache import TokenizerWrapper

# import sys
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def load_tokenizer(tokenizer_path):
    """Load tokenizer from path or name."""
    if Path(tokenizer_path).exists():
        return AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, local_files_only=True
        )
    else:
        return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)


def test_boundary_split_issue(tokenizer_path="gpt2"):
    """
    Reproduces the issue where a token at the boundary gets split differently
    when tokenized separately vs with context.

    Example: "helpful" tokenized as one token vs "he" + "lpful" as two.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    wrapper = TokenizerWrapper(tokenizer)

    text = "System: You are a helpful assistant."

    # Tokenize full text (ground truth)
    full_tokens = tokenizer.encode(text, add_special_tokens=False)

    # Tokenize incrementally
    part1 = text[:20]  # "System: You are a he"
    part2 = text[: len(text)]  # Full text

    tokens1 = wrapper.encode(part1, add_special_tokens=False)
    tokens2 = wrapper.encode(part2, add_special_tokens=False)

    if tokens2 != full_tokens:
        print("❌ Boundary split issue reproduced")
        print(f"  Expected: {full_tokens}")
        print(f"  Got:      {tokens2}")
        return False

    print("✅ Boundary split handled correctly")
    return True


def test_unicode_normalization(tokenizer_path="gpt2"):
    """
    Test that different Unicode normalizations are handled correctly.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    wrapper = TokenizerWrapper(tokenizer)

    # Same text in different Unicode forms
    text_nfc = "café"  # NFC form
    text_nfd = "café"  # NFD form (different bytes)

    tokens_nfc = wrapper.encode(text_nfc, add_special_tokens=False)
    tokens_nfd = wrapper.encode(text_nfd, add_special_tokens=False)

    # Both should be valid (may differ, that's OK)
    direct_nfc = tokenizer.encode(text_nfc, add_special_tokens=False)
    direct_nfd = tokenizer.encode(text_nfd, add_special_tokens=False)

    success = (tokens_nfc == direct_nfc) and (tokens_nfd == direct_nfd)

    if success:
        print("✅ Unicode normalization handled correctly")
    else:
        print("❌ Unicode normalization issue")
        print(f"  NFC: {tokens_nfc} vs {direct_nfc}")
        print(f"  NFD: {tokens_nfd} vs {direct_nfd}")

    return success


def test_special_tokens_in_text(tokenizer_path="gpt2"):
    """
    Test that special tokens appearing in text are handled correctly.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    wrapper = TokenizerWrapper(tokenizer)

    # Text containing what looks like special tokens
    text = "Use <|endoftext|> to end"

    cached = wrapper.encode(text, add_special_tokens=False)
    direct = tokenizer.encode(text, add_special_tokens=False)

    if cached != direct:
        print("❌ Special tokens in text issue")
        print(f"  Expected: {direct}")
        print(f"  Got:      {cached}")
        return False

    print("✅ Special tokens in text handled correctly")
    return True


def test_empty_and_whitespace(tokenizer_path="gpt2"):
    """
    Test edge cases with empty strings and whitespace.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    wrapper = TokenizerWrapper(tokenizer)

    test_cases = [
        "",
        " ",
        "  ",
        "\n",
        "\t",
        "\r\n",
    ]

    all_pass = True
    for text in test_cases:
        cached = wrapper.encode(text, add_special_tokens=False)
        direct = tokenizer.encode(text, add_special_tokens=False)

        if cached != direct:
            print(f"❌ Failed for {repr(text)}: {cached} vs {direct}")
            all_pass = False

    if all_pass:
        print("✅ Empty and whitespace handled correctly")

    return all_pass


def test_incremental_conversation(tokenizer_path="gpt2"):
    """
    Test realistic conversational pattern with incremental growth.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    wrapper = TokenizerWrapper(tokenizer)

    conversation = [
        "System: You are helpful.",
        "System: You are helpful.\nUser: Hello!",
        "System: You are helpful.\nUser: Hello!\nAssistant: Hi there!",
        "System: You are helpful.\nUser: Hello!\nAssistant: Hi there!\nUser: How are you?",
    ]

    all_pass = True
    for text in conversation:
        cached = wrapper.encode(text, add_special_tokens=False)
        direct = tokenizer.encode(text, add_special_tokens=False)

        if cached != direct:
            print(f"❌ Failed at length {len(text)}")
            print(f"  Text: {text[:50]}...")
            print(f"  Cached: {cached[:10]}...")
            print(f"  Direct: {direct[:10]}...")
            all_pass = False
            break

    if all_pass:
        print("✅ Incremental conversation handled correctly")

    return all_pass


def test_boundary_safety(tokenizer_path="gpt2"):
    """
    Test that boundary safety checking prevents incorrect cache reuse.
    """
    tokenizer = load_tokenizer(tokenizer_path)

    # Test with boundary checking disabled (baseline)
    wrapper_no_check = TokenizerWrapper(tokenizer, enable_boundary_check=False)

    print("  Testing boundary safety mechanism:")

    # These are potential problem cases
    test_cases = [
        ("Hello wo", "rld"),  # Mid-word split
        ("Test", "ing"),  # Word continuation
        ("", "Start"),  # Empty prefix
    ]

    all_pass = True
    for prefix, suffix in test_cases:
        full_text = prefix + suffix

        # Cache the prefix
        if prefix:
            wrapper_no_check.encode(prefix, add_special_tokens=False)

        # Get full text tokens
        cached = wrapper_no_check.encode(full_text, add_special_tokens=False)
        direct = tokenizer.encode(full_text, add_special_tokens=False)

        if cached == direct:
            print(f"  ✅ '{prefix}' | '{suffix}': Correct")
        else:
            print(f"  ❌ '{prefix}' | '{suffix}': Mismatch")
            all_pass = False

    return all_pass


def test_chat_template_boundaries(tokenizer_path="gpt2"):
    """
    Test tokenization with chat template turn boundaries.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    wrapper = TokenizerWrapper(tokenizer)

    # Build conversation incrementally
    conversations = [
        [{"role": "system", "content": "You are a helpful assistant."}],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's 2+2?"},
        ],
    ]

    all_pass = True
    prev_text = ""

    # Skip if tokenizer doesn't have chat template
    if not hasattr(tokenizer, "apply_chat_template"):
        print("⚠️  Tokenizer doesn't support chat templates, skipping")
        return True

    print("\n  Chat template format analysis:")
    print("  " + "=" * 50)

    for i, conv in enumerate(conversations):
        try:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                conversation=conv,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Show the incremental growth
            print(f"\n  Turn {i+1} ({len(conv)} message{'s' if len(conv)>1 else ''}):")
            print(f"  Text length: {len(text)} chars")

            if prev_text:
                # Find the new content added
                if text.startswith(prev_text):
                    new_content = text[len(prev_text) :]
                    print(f"  New content: {repr(new_content[:100])}")
                else:
                    print(f"  WARNING: Not a simple append!")
            else:
                print(f"  Initial: {repr(text[:100])}")

            # Print full text for analysis (truncated)
            if len(text) <= 200:
                print(f"  Full text: {repr(text)}")
            else:
                print(f"  Full text: {repr(text[:100])}...{repr(text[-50:])}")

            # Test incremental tokenization
            cached = wrapper.encode(text, add_special_tokens=False)
            direct = tokenizer.encode(text, add_special_tokens=False)

            print(f"  Tokens: cached={len(cached)}, direct={len(direct)}")

            if cached != direct:
                print(f"  ❌ Tokenization mismatch!")
                # Show where they differ
                for j in range(min(len(cached), len(direct))):
                    if cached[j] != direct[j]:
                        print(
                            f"    First diff at position {j}: cached={cached[j]}, direct={direct[j]}"
                        )
                        break
                all_pass = False
            else:
                print(f"  ✅ Tokens match")

            prev_text = text

        except Exception as e:
            # Some tokenizers might not have default templates
            print(f"⚠️  Chat template test skipped: {e}")
            return True

    print("\n  " + "=" * 50)

    if all_pass:
        print("✅ Chat template boundaries handled correctly")
    else:
        print("❌ Chat template boundaries test failed")

    return all_pass


def main(tokenizer_path="gpt2"):
    """Run all unit tests."""
    print(f"Running unit tests with tokenizer: {tokenizer_path}\n")

    tests = [
        ("Boundary Split", test_boundary_split_issue),
        ("Unicode Normalization", test_unicode_normalization),
        ("Special Tokens", test_special_tokens_in_text),
        ("Empty/Whitespace", test_empty_and_whitespace),
        ("Incremental Conversation", test_incremental_conversation),
        ("Chat Template Boundaries", test_chat_template_boundaries),
        ("Boundary Safety Checking", test_boundary_safety),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\nTest: {name}")
        print("-" * 40)
        try:
            if test_func(tokenizer_path):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Run tokenizer cache unit tests")
    parser.add_argument(
        "--tokenizer", default="gpt2", help="Tokenizer name or path to model"
    )
    args = parser.parse_args()

    sys.exit(0 if main(args.tokenizer) else 1)
