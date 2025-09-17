from enum import IntEnum, auto

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as SGLANG_SpeculativeAlgorithm

class SpeculativeAlgorithm(IntEnum):

    NONE = SGLANG_SpeculativeAlgorithm.NONE
    EAGLE = SGLANG_SpeculativeAlgorithm.EAGLE
    EAGLE3 = SGLANG_SpeculativeAlgorithm.EAGLE3
    STANDALONE = SGLANG_SpeculativeAlgorithm.STANDALONE
    PHOENIX = auto()
        
    def is_none(self):
        return self == SGLANG_SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self == SGLANG_SpeculativeAlgorithm.EAGLE or self == SGLANG_SpeculativeAlgorithm.EAGLE3

    def is_eagle3(self):
        return self == SGLANG_SpeculativeAlgorithm.EAGLE3

    def is_standalone(self):
        return self == SGLANG_SpeculativeAlgorithm.STANDALONE

    def is_phoenix(self):
        return self == SpeculativeAlgorithm.PHOENIX

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SGLANG_SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SGLANG_SpeculativeAlgorithm.EAGLE3,
            "PHOENIX": SpeculativeAlgorithm.PHOENIX,
            "STANDALONE": SGLANG_SpeculativeAlgorithm.STANDALONE,
            None: SGLANG_SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
