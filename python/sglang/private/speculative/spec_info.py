from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Type, Union

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker


class SpeculativeAlgorithm(Enum):
    """Enumeration of speculative decoding algorithms."""

    EAGLE = auto()
    EAGLE3 = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()

    # =============================================================================
    PHOENIX = auto()
    PHOENIX2 = auto()
    # =============================================================================

    @classmethod
    def from_string(cls, name: Optional[str]) -> SpeculativeAlgorithm:
        if name is None:
            return cls.NONE
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown speculative algorithm name: {name}")

    def is_none(self) -> bool:
        return self == SpeculativeAlgorithm.NONE

    # =============================================================================
    def is_eagle(self) -> bool:
        # NOTE: EAGLE3 is a variant of EAGLE
        return (
            self == SpeculativeAlgorithm.EAGLE
            or self == SpeculativeAlgorithm.EAGLE3
            or self == SpeculativeAlgorithm.PHOENIX
        )

    # =============================================================================

    def is_eagle3(self) -> bool:
        return self == SpeculativeAlgorithm.EAGLE3

    def is_standalone(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    # =============================================================================
    def is_phoenix(self) -> bool:
        return (
            self == SpeculativeAlgorithm.PHOENIX
            or self == SpeculativeAlgorithm.PHOENIX2
        )

    def is_phoenix2(self) -> bool:
        return self == SpeculativeAlgorithm.PHOENIX2

    # =============================================================================

    def supports_spec_v2(self) -> bool:
        return self.is_eagle() or self.is_standalone()

    def create_worker(
        self, enable_overlap: bool = False
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        if self.is_none():
            return None

        # =============================================================================
        if self.is_phoenix() or self.is_phoenix2():
            from sglang.private.speculative.phoenix_worker import PhoenixWorker

            return PhoenixWorker
        # =============================================================================
        elif self.is_eagle():
            if enable_overlap:
                from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

                return EAGLEWorkerV2

            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            return EAGLEWorker
        elif self.is_standalone():
            if enable_overlap:
                from sglang.srt.speculative.standalone_worker_v2 import (
                    StandaloneWorkerV2,
                )

                return StandaloneWorkerV2

            from sglang.srt.speculative.standalone_worker import StandaloneWorker

            return StandaloneWorker
        elif self.is_ngram():
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support overlap worker creation."
                )

            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("Unreachable code path in create_worker.")


__all__ = ["SpeculativeAlgorithm"]
