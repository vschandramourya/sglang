from typing import Optional

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ToolChoice
from sglang.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat as SGLangOpenAIServingChat,
)


class OpenAIServingChat(SGLangOpenAIServingChat):

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
