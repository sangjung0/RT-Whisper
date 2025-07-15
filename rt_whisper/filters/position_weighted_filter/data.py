from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from dataclasses import dataclass

if TYPE_CHECKING:
    from rt_whisper.data import TokenState, Token


@dataclass(slots=True)
class PositionWeightedFilterParam:
    offset: int
    chunk: np.ndarray
    segment_tokens: list[Token]

    @staticmethod
    def from_context(state: TokenState) -> "PositionWeightedFilterParam":
        return PositionWeightedFilterParam(
            offset=state.offset,
            chunk=state.chunk,
            segment_tokens=state.segment_tokens,
        )


@dataclass(slots=True)
class PositionWeightedFilterResult:
    segment_tokens: list[Token]

    def update_context(self, state: TokenState) -> None:
        state.segment_tokens = self.segment_tokens
