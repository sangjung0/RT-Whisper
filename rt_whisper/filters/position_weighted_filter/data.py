from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from dataclasses import dataclass

if TYPE_CHECKING:
    from rt_whisper.data import TokenState, Token
    from rt_whisper.processors.asr.data import ASRState


@dataclass(slots=True)
class PositionWeightedFilterParam:
    offset: int
    merged_chunk: np.ndarray
    segment_tokens: list[Token]

    @staticmethod
    def from_context(state: TokenState, asr_state: ASRState) -> "PositionWeightedFilterParam":
        return PositionWeightedFilterParam(
            offset=state.offset,
            merged_chunk=asr_state.merged_chunk,
            segment_tokens=state.segment_tokens,
        )


@dataclass(slots=True)
class PositionWeightedFilterResult:
    segment_tokens: list[Token]

    def update_context(self, state: TokenState) -> None:
        state.segment_tokens = self.segment_tokens
