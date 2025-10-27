from __future__ import annotations

from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    register_speculative_algorithm,
)


def _create_phoenix_worker(**kwargs):
    from sglang.private.speculative.phoenix_worker import PhoenixWorker

    return PhoenixWorker(**kwargs)


phoenix_algorithm = register_speculative_algorithm(
    "PHOENIX",
    _create_phoenix_worker,
    flags=("EAGLE",),
)

if not hasattr(SpeculativeAlgorithm, "is_phoenix"):

    def _is_phoenix(self: SpeculativeAlgorithm) -> bool:
        return self == phoenix_algorithm

    SpeculativeAlgorithm.is_phoenix = _is_phoenix  # type: ignore[attr-defined]


__all__ = ["SpeculativeAlgorithm"]
