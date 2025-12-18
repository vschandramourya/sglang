"""Private HarmonyParser with reset() method to prevent memory leaks."""

from sglang.srt.parser.harmony_parser import HarmonyParser as BaseHarmonyParser


class HarmonyParser(BaseHarmonyParser):
    """
    Extended HarmonyParser with reset() method.

    This prevents state leakage between requests when the parser instance
    is reused across multiple parsing operations.
    """

    def reset(self):
        """Reset parser state for reuse. Call this between requests."""
        self.strategy = None
        self._buffer = ""
        self._should_filter_commentary = False
        self._partial_commentary = ""
