"""
Private GptOssDetector with fixes for:
1. Function name extraction (handles "functions." prefix correctly)
2. Case-insensitive enum value normalization
3. Memory leak prevention via reset() method
"""

import json
import logging
from typing import List, Optional

from sglang.private.parser.harmony_parser import HarmonyParser
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.gpt_oss_detector import (
    GptOssDetector as BaseGptOssDetector,
)

logger = logging.getLogger(__name__)


class GptOssDetector(BaseGptOssDetector):
    """
    Fixed GptOssDetector that properly handles:
    1. Tool names containing dots (e.g., "patient.get_mri_report")
    2. Case-insensitive enum value matching
    3. State reset between requests to prevent memory leaks
    """

    def __init__(self):
        super().__init__()
        # Use private HarmonyParser with reset() support
        self.harmony_parser = HarmonyParser()

    def reset(self):
        """Reset detector state for reuse. Call this between requests."""
        # Reset base class streaming state
        self._buffer = ""
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        # Reset HarmonyParser
        self.harmony_parser.reset()
        # Clear any cached state
        if hasattr(self, "_tool_indices"):
            delattr(self, "_tool_indices")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse TypeScript-style function calls from complete text."""
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        # Reset HarmonyParser for clean one-shot parsing (prevent state leakage)
        self.harmony_parser.reset()

        # Parse with HarmonyParser
        events = self.harmony_parser.parse(text)
        # Flush buffer for complete parsing
        events += self.harmony_parser.parse("")

        tool_indices = self._get_tool_indices(tools)
        calls = []
        normal_parts = []
        tool_index = 0

        for event in events:
            if event.event_type == "tool_call":
                # Extract tool call from event content
                tool_call = self._extract_tool_call_from_event(
                    event.raw_text if event.raw_text else event.content,
                    tool_indices,
                    tool_index,
                    tools,
                )
                if tool_call:
                    calls.append(tool_call)
                    tool_index += 1
            elif event.event_type == "normal":
                normal_parts.append(event.content)
            # Ignore reasoning events in function call context

        normal_text = " ".join(normal_parts).strip()
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Parse incremental streaming text for TypeScript-style function calls."""
        self._buffer += new_text

        # Always use HarmonyParser for parsing to ensure proper filtering
        events = self.harmony_parser.parse(new_text)

        # If there are no parsed events and the chunk contains no Harmony structural
        # markers, treat it as plain text and pass it through.
        if not events:
            has_harmony_markers = any(
                marker in self._buffer
                for marker in (
                    "<|start|>",
                    "<|channel|>",
                    "<|message|>",
                    "<|constrain|>",
                    "<|end|>",
                    "<|call|>",
                    "<|return|>",
                    "assistantfinal",
                )
            )
            if not has_harmony_markers:
                out = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=out, calls=[])

        # Quick check if we might have tool calls
        if (
            "<|channel|>commentary to=" not in self._buffer
            and not self.current_tool_name_sent
        ):
            if (
                "<|channel|>final" in self._buffer
                or "assistantfinal" in self._buffer.lower()
            ):
                normal_text = "".join(
                    [e.content for e in events if e.event_type == "normal"]
                )
                if normal_text:
                    self._buffer = ""
                    return StreamingParseResult(normal_text=normal_text, calls=[])

            normal_text = "".join(
                [e.content for e in events if e.event_type == "normal"]
            )
            if normal_text or events:
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text, calls=[])
            else:
                return StreamingParseResult(normal_text="", calls=[])

        if not events:
            return StreamingParseResult(normal_text="", calls=[])

        # Initialize state if needed
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls = []
        normal_text = ""

        for event in events:
            if event.event_type == "tool_call":
                tool_call_info = self._extract_tool_call_from_event(
                    event.raw_text if event.raw_text else event.content,
                    self._tool_indices,
                    self.current_tool_id if self.current_tool_id >= 0 else 0,
                    tools,
                )

                if tool_call_info:
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]

                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": tool_call_info.name,
                        "arguments": json.loads(tool_call_info.parameters),
                    }

                    calls.append(tool_call_info)

                    self.streamed_args_for_tool[self.current_tool_id] = (
                        tool_call_info.parameters
                    )

                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

            elif event.event_type == "normal":
                normal_text += event.content

        self._buffer = ""

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _extract_tool_call_from_event(
        self, content: str, tool_indices: dict, tool_index: int, tools: List[Tool]
    ) -> Optional[ToolCallItem]:
        """
        Extract tool call information from HarmonyParser event content.

        Content format: "commentary to=functions.get_weather<|constrain|>json<|message|>{...}"

        Fixed to handle tool names that contain dots (e.g., "patient.get_mri_report").
        """
        match = self.tool_extract_pattern.search(content)

        if not match:
            logger.debug(f"Could not extract tool call from: {content[:100]}")
            return None

        full_function_name = match.group(1)
        json_content = match.group(2)

        # Extract function name (strip "functions." prefix if present)
        # The model outputs "functions.{tool_name}" where tool_name can itself contain dots
        # (e.g., "functions.patient.get_mri_report" -> "patient.get_mri_report")
        if full_function_name.startswith("functions."):
            function_name = full_function_name[len("functions.") :]
        elif "." in full_function_name:
            function_name = full_function_name.split(".")[-1]
        else:
            function_name = full_function_name

        # Check if tool exists
        if function_name not in tool_indices:
            logger.debug(f"Function {function_name} not in available tools")
            if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return None  # Skip unknown tools (default legacy behavior)

        # Parse JSON arguments
        try:
            arguments = json.loads(json_content) if json_content.strip() else {}
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON arguments: {e}")
            return None

        # Normalize enum values (case-insensitive matching)
        arguments = self._normalize_enum_values(arguments, function_name, tools)

        return ToolCallItem(
            tool_index=tool_index,
            name=function_name,
            parameters=json.dumps(arguments, ensure_ascii=False),
        )

    def _normalize_enum_values(
        self, arguments: dict, function_name: str, tools: List[Tool]
    ) -> dict:
        """
        Normalize argument values to match enum case in the tool schema.
        For example, if schema has enum ["Patio"] and argument has "patio",
        replace with "Patio".
        """
        # Find the tool schema - handle both Tool objects and dicts
        tool_schema = None
        for tool in tools:
            try:
                # Handle Tool object
                if hasattr(tool, "function"):
                    func = tool.function
                    tool_name = func.name if hasattr(func, "name") else func.get("name")
                    if tool_name == function_name:
                        tool_schema = (
                            func.parameters
                            if hasattr(func, "parameters")
                            else func.get("parameters")
                        )
                        break
                # Handle dict
                elif isinstance(tool, dict) and "function" in tool:
                    func = tool["function"]
                    if func.get("name") == function_name:
                        tool_schema = func.get("parameters")
                        break
            except Exception:
                continue

        if not tool_schema:
            return arguments

        # Get properties - handle both dict and object access
        properties = None
        if isinstance(tool_schema, dict):
            properties = tool_schema.get("properties")
        elif hasattr(tool_schema, "properties"):
            properties = tool_schema.properties

        if not properties:
            return arguments

        for arg_name, arg_value in arguments.items():
            # Get property schema
            prop_schema = None
            if isinstance(properties, dict):
                prop_schema = properties.get(arg_name)
            elif hasattr(properties, arg_name):
                prop_schema = getattr(properties, arg_name)

            if not prop_schema:
                continue

            # Get type and items - handle both dict and object
            prop_type = (
                prop_schema.get("type")
                if isinstance(prop_schema, dict)
                else getattr(prop_schema, "type", None)
            )
            prop_enum = (
                prop_schema.get("enum")
                if isinstance(prop_schema, dict)
                else getattr(prop_schema, "enum", None)
            )
            prop_items = (
                prop_schema.get("items")
                if isinstance(prop_schema, dict)
                else getattr(prop_schema, "items", None)
            )

            # Handle direct enum on property
            if prop_enum and isinstance(arg_value, str):
                arguments[arg_name] = self._match_enum_value(arg_value, prop_enum)

            # Handle array of enums
            elif prop_type == "array" and isinstance(arg_value, list) and prop_items:
                items_enum = (
                    prop_items.get("enum")
                    if isinstance(prop_items, dict)
                    else getattr(prop_items, "enum", None)
                )
                if items_enum:
                    arguments[arg_name] = [
                        (
                            self._match_enum_value(v, items_enum)
                            if isinstance(v, str)
                            else v
                        )
                        for v in arg_value
                    ]

        return arguments

    def _match_enum_value(self, value: str, enum_values: list) -> str:
        """Match value to enum case-insensitively. Returns original if no match."""
        value_lower = value.lower()
        for enum_val in enum_values:
            if isinstance(enum_val, str) and enum_val.lower() == value_lower:
                return enum_val
        return value

    def supports_structural_tag(self) -> bool:
        """GptOssDetector does not support structural tag format."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError("structure_info not used with HarmonyParser")
