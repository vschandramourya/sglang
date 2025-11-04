#!/usr/bin/env python3
"""
Simple script to send synthetic multi-turn chat requests to OpenAI-compatible server.
For unit testing, simulates multi-turn conversation requests.
"""

import argparse
import json
import random
import sys
import time
from typing import Any, Dict, List

try:
    import requests
except ImportError:
    print("Error: requests library required. Install: pip install requests")
    sys.exit(1)

from sglang.private.inc_tokenizer.tests.benchmark import (
    generate_conversation_turn,
    generate_synthetic_conversations,
)


def send_request(
    server_url: str, payload: Dict[str, Any], timeout: float = 30.0
) -> Dict:
    """Send a single request to the server"""
    if not server_url.endswith("/v1/chat/completions"):
        if server_url.endswith("/"):
            server_url = server_url + "v1/chat/completions"
        else:
            server_url = server_url + "/v1/chat/completions"

    try:
        response = requests.post(
            server_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Send synthetic multi-turn chat requests"
    )
    parser.add_argument("--server", default="http://localhost:12345", help="Server URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--num-convs", type=int, default=3, help="Number of conversations"
    )
    parser.add_argument(
        "--max-turns", type=int, default=5, help="Max turns per conversation"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    args = parser.parse_args()

    # Generate conversations
    conversations = generate_synthetic_conversations(args.num_convs, "short")

    print(f"Sending {len(conversations)} conversations to {args.server}...")

    success_count = 0
    error_count = 0

    for conv_idx, conversation in enumerate(conversations):
        # Limit number of turns
        conversation = conversation[: args.max_turns * 2]  # user + assistant pairs

        # Incremental sending: simulate multi-turn conversations
        for turn in range(1, len(conversation) + 1):
            messages = conversation[:turn]

            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 100,
            }

            response = send_request(args.server, payload, args.timeout)

            if "error" in response:
                error_count += 1
                print(f"  Conv {conv_idx+1} Turn {turn}: ERROR - {response['error']}")
            else:
                success_count += 1
                if turn == len(conversation):
                    print(f"  Conv {conv_idx+1}: OK ({turn} turns)")

    print(f"\nTotal: {success_count} success, {error_count} errors")

    # Return non-zero exit code if there are errors
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
