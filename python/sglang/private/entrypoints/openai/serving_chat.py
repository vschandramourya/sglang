"""
Private OpenAIServingChat that uses the fixed GptOssDetector for tool call parsing.
"""

import logging
from typing import Dict, List, Optional

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
    ToolChoice,
)
from sglang.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat as SGLangOpenAIServingChat,
)

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
        return super()._apply_jinja_template(request, tools, is_multimodal)
