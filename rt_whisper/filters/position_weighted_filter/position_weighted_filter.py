from __future__ import annotations
from typing import TYPE_CHECKING

from rt_whisper.abstracts import Worker

from .data import PositionWeightedFilterParam, PositionWeightedFilterResult
from .service import *

if TYPE_CHECKING:
    from rt_whisper.data import TokenState


class PositionWeightedFilter(Worker):
    def __init__(
        self,
        boundary: float,
    ):
        super().__init__()
        self.__BOUNDARY = boundary

    # override
    def _can_process(self, context: TokenState):
        if context.chunk.shape[0] > 0:
            return PositionWeightedFilterParam.from_context(context)
        return None

    # override
    def _process(
        self, param: PositionWeightedFilterParam
    ) -> PositionWeightedFilterResult:

        tokens = filter_by_position_weighted(
            param.segment_tokens,
            param.offset,
            param.chunk.shape[0],
            self.__BOUNDARY,
        )

        return PositionWeightedFilterResult(
            # tokens=param.merged_candidate_tokens
        )

    # override
    def _update(
        self, context: TokenState, result: PositionWeightedFilterResult
    ) -> None:
        result.update_context(context)
