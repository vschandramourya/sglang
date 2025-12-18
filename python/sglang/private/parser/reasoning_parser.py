"""Private GptOssDetector for reasoning parsing with reset() method."""

from sglang.private.parser.harmony_parser import HarmonyParser
from sglang.srt.parser.reasoning_parser import GptOssDetector as BaseGptOssDetector
from sglang.srt.parser.reasoning_parser import StreamingParseResult


class GptOssDetector(BaseGptOssDetector):
    """
    Extended GptOssDetector for reasoning parsing with reset() method.

    This prevents state leakage between requests when the detector instance
    is reused across multiple parsing operations.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        super().__init__(
            stream_reasoning=stream_reasoning,
            force_reasoning=force_reasoning,
        )
        # Use private HarmonyParser with reset() support
        self.parser = HarmonyParser()

    def reset(self):
        """Reset detector state for reuse. Call this between requests."""
        self.parser.reset()

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        # Reset parser for clean one-shot parsing (prevent state leakage)
        self.parser.reset()

        events = self.parser.parse(text)
        # Flush the buffer for one-shot parsing
        events += self.parser.parse("")

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )
