"""
Private KimiK2Detector with fixes for:
1. Structural tag grammar issues (disabled as workaround)
2. Memory leak prevention via reset() method
3. Improved state management for streaming
4. Support for hyphens in function names (e.g., agent__hf-search)

This module provides an improved version of the KimiK2Detector that:
- Disables structural_tag support to avoid grammar compilation issues in xgrammar
- Adds reset() method to prevent memory leaks between requests
- Improves streaming state management
- Fixes regex patterns to support hyphens in function names
"""

import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.kimik2_detector import (
    KimiK2Detector as BaseKimiK2Detector,
)
from sglang.srt.function_call.utils import _is_complete_json

logger = logging.getLogger(__name__)


class KimiK2Detector(BaseKimiK2Detector):
    """
    Fixed KimiK2Detector that properly handles:
    1. Structural tag grammar issues (disabled as workaround)
    2. State reset between requests to prevent memory leaks
    3. Improved streaming state management
    4. Hyphens in function names (e.g., agent__hf-search)

    Format Structure:
    ```
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{func_name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    <|tool_calls_section_end|>
    ```

    Note: Function names can contain letters, digits, underscores, dots, and hyphens.
    """

    def __init__(self):
        super().__init__()

        # Override regex patterns to fix multiple issues:
        # 1. Support hyphens in function names (e.g., "agent__hf-search")
        # 2. Support newlines in JSON arguments (re.DOTALL)
        # 3. Support concatenated tool calls (use [^<]+ instead of .+ to prevent
        #    matching across <| token boundaries)
        #
        # Key pattern choices:
        # - [^<]+ for tool_call_id: matches any char except '<', preventing overflow
        #   into next token boundary
        # - .*? (non-greedy) with DOTALL: matches JSON including newlines, stops at
        #   first <|tool_call_end|>
        # - \s* between tokens: handles variable spacing

        # Pattern for complete tool calls (one-shot parsing)
        # Handles: concatenated calls, newlines in JSON, angle brackets in JSON values
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        # Pattern for streaming partial tool calls
        # Uses [^<]+ to prevent matching across token boundaries
        self.stream_tool_call_portion_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)",
            re.DOTALL,
        )

        # Pattern for parsing tool call IDs like "functions.agent__hf-search:0"
        # Supports: letters, digits, underscores, dots, hyphens in function names
        self.tool_call_id_regex = re.compile(
            r"^(?:functions\.)?(?P<name>[^:]+):(?P<index>\d+)$"
        )

    def reset(self):
        """Reset detector state for reuse. Call this between requests."""
        # Reset base class streaming state
        self._buffer = ""
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        # Reset Kimi K2 specific state
        self._last_arguments = ""
        # Clear any cached state
        if hasattr(self, "_tool_indices"):
            delattr(self, "_tool_indices")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        Resets state before parsing to prevent state leakage from previous requests.
        """
        # Reset state for clean one-shot parsing
        self.reset()
        return super().detect_and_parse(text, tools)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for KimiK2 format.

        This is an improved version that handles edge cases better.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        has_tool_call = (
            self.bot_token in current_text or self.tool_call_start_token in current_text
        )

        if not has_tool_call:
            self._buffer = ""
            for e_token in [self.eot_token, self.tool_call_end_token]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            match = self.stream_tool_call_portion_regex.search(current_text)
            if match:
                function_id = match.group("tool_call_id")
                function_args = match.group("function_arguments")

                m = self.tool_call_id_regex.match(function_id)
                if not m:
                    logger.warning("Unexpected tool_call_id format: %s", function_id)
                    return StreamingParseResult(normal_text="", calls=calls)
                function_name = m.group("name")

                # Validate function name against available tools
                if function_name not in self._tool_indices:
                    logger.warning(
                        "Unknown function name: %s, available: %s",
                        function_name,
                        list(self._tool_indices.keys()),
                    )
                    # Continue processing but log the warning

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=function_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    # Store the tool call info for serving layer completions endpoint
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": function_name,
                        "arguments": {},
                    }
                else:
                    argument_diff = (
                        function_args[len(self._last_arguments) :]
                        if function_args.startswith(self._last_arguments)
                        else function_args
                    )

                    parsed_args_diff = argument_diff.split("<|tool_call_end|>", 1)[0]

                    if parsed_args_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=parsed_args_diff,
                            )
                        )
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += parsed_args_diff

                    parsed_args = function_args.split("<|tool_call_end|>", 1)[0]
                    if _is_complete_json(parsed_args):
                        try:
                            parsed_args_obj = json.loads(parsed_args)
                            self.prev_tool_call_arr[self.current_tool_id][
                                "arguments"
                            ] = parsed_args_obj
                        except json.JSONDecodeError:
                            pass

                        # Find the end of the current tool call and remove only that part from buffer
                        tool_call_end_pattern = (
                            r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>"
                        )
                        end_match = re.search(
                            tool_call_end_pattern, current_text, re.DOTALL
                        )
                        if end_match:
                            # Remove the completed tool call from buffer, keep any remaining content
                            self._buffer = current_text[end_match.end() :]
                        else:
                            self._buffer = ""

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def supports_structural_tag(self) -> bool:
        """Return False to disable structural tag format.

        Structural tag support is disabled for Kimi K2 due to grammar compilation
        issues with xgrammar. The grammar format is correct but fails in the
        server context due to potential TP synchronization or caching issues.

        To re-enable, change this to return True. The structure_info() method
        is already properly implemented.
        """
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Return function that creates StructureInfo for guided generation.

        Kimi K2 format:
        <|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{idx}<|tool_call_argument_begin|>{json}<|tool_call_end|><|tool_calls_section_end|>

        The {idx} is the call index (0 for first call, 1 for second, etc.).

        For constrained generation, we use a common trigger that fires when
        the model starts a tool call section. The grammar then branches based
        on the tool name in the begin pattern.

        Note: This is currently disabled via supports_structural_tag() returning False.
        """

        def get_info(name: str) -> StructureInfo:
            # Common trigger shared by all tools - fires at start of tool call
            # This is compatible with xgrammar's structural_tag requirements
            common_trigger = f"{self.bot_token}{self.tool_call_start_token}functions."

            return StructureInfo(
                # Begin: full prefix including tool name, index, and argument delimiter
                begin=f"{self.bot_token}{self.tool_call_start_token}functions.{name}:0<|tool_call_argument_begin|>",
                # End: close the call and section
                end=f"{self.tool_call_end_token}{self.eot_token}",
                # Trigger: common prefix for all tools (branches at tool name)
                trigger=common_trigger,
            )

        return get_info
