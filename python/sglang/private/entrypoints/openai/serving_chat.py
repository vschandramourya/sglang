"""
Private OpenAIServingChat that uses the fixed GptOssDetector for tool call parsing.
"""

import logging
from typing import Dict, List, Optional

import orjson

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
    ToolChoice,
)
from sglang.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat as SGLangOpenAIServingChat,
)
from sglang.srt.function_call.core_types import ToolCallItem

logger = logging.getLogger(__name__)


def _preprocess_tools_for_gpt_oss(
    tools: Optional[List[Dict]],
) -> Optional[List[Dict]]:
    """Preprocess tools for GPT-OSS to ensure description is not None."""
    if tools is None:
        return None
    processed = []
    for tool in tools:
        tool_copy = tool.copy()
        if tool_copy.get("description") is None:
            tool_copy["description"] = ""
        processed.append(tool_copy)
    return processed


class OpenAIServingChat(SGLangOpenAIServingChat):
    """
    Private OpenAIServingChat that uses the fixed FunctionCallParser
    with the private GptOssDetector.
    """

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:

        if result := super()._validate_request(request):
            return result

        required_tool_choice = (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
        ) or isinstance(request.tool_choice, ToolChoice)

        if (
            required_tool_choice
            and self.tokenizer_manager.server_args.tool_choice_mode == "disabled"
        ):
            return "Tool choice is set to required but not supported."

        if (
            required_tool_choice
            and self.tokenizer_manager.server_args.tool_choice_mode == "ignored"
        ):
            request.tool_choice = "auto"

        return None

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply Jinja chat template with GPT-OSS tool preprocessing."""
        # Preprocess tools for GPT-OSS to ensure required fields have defaults
        if self.is_gpt_oss:
            tools = _preprocess_tools_for_gpt_oss(tools)

        # Process tool_calls in previous messages to align their IDs
        # This is especially important for Kimi-K2 which uses a specific ID format
        history_tool_calls_cnt = 0
        for message in request.messages:
            # Only process assistant messages with tool_calls
            if message.role == "assistant" and message.tool_calls:
                for tool_call in message.tool_calls:
                    # Get the original tool index from the ID if possible
                    # For new tool calls without proper IDs, use the position in the list
                    tool_index = message.tool_calls.index(tool_call)

                    # Create a ToolCallItem to pass to _process_tool_call_id
                    # Need to extract arguments as a JSON string
                    arguments_str = tool_call.function.arguments
                    if isinstance(arguments_str, dict):
                        arguments_str = orjson.dumps(arguments_str).decode("utf-8")

                    call_item = ToolCallItem(
                        tool_index=tool_index,
                        name=tool_call.function.name,
                        parameters=arguments_str or "{}",
                    )

                    # Generate the aligned tool_call_id
                    aligned_id = self._process_tool_call_id(
                        call_item, history_tool_calls_cnt
                    )

                    # Update the tool call ID in place
                    tool_call.id = aligned_id

                    logger.debug(
                        f"Aligned tool_call ID in message history: {aligned_id}, "
                        f"tool_name: {tool_call.function.name}, "
                        f"history_cnt: {history_tool_calls_cnt}"
                    )

                # Increment the counter by the number of tool calls in this message
                history_tool_calls_cnt += len(message.tool_calls)

        return super()._apply_jinja_template(request, tools, is_multimodal)
