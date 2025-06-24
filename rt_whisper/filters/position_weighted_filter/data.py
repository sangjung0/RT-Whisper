from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

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
    # tokens: list[Token]

    def update_context(self, state: TokenState) -> None:
        # context.merged_candidate_tokens = self.tokens
        pass
