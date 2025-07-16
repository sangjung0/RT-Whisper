from __future__ import annotations
from typing import TYPE_CHECKING

from rt_whisper.abstracts import Worker
from rt_whisper.filters.position_weighted_filter.data import (
    PositionWeightedFilterParam,
    PositionWeightedFilterResult,
)
from rt_whisper.filters.position_weighted_filter.service import (
    filter_by_position_weighted,
)

if TYPE_CHECKING:
    from rt_whisper import RTWhisperLogger
    from rt_whisper.data import TokenState


class PositionWeightedFilter(Worker):
    def __init__(self, boundary: float, logger: RTWhisperLogger):
        super().__init__()
        self.logger = logger
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
        self.logger.debug(f"Processing Position Weighted Filter", group_level=1)

        self.logger.debug(
            f"Before: {''.join(str(t) for t in param.segment_tokens if t.is_word)}", group_level=2
        )
        tokens = filter_by_position_weighted(
            param.segment_tokens,
            param.offset,
            param.chunk.shape[0],
            self.__BOUNDARY,
        )
        self.logger.debug(f"After: {''.join(str(t) for t in tokens if t.is_word)}", group_level=2)

        return PositionWeightedFilterResult(
            segment_tokens=tokens,
        )

    # override
    def _update(
        self, context: TokenState, result: PositionWeightedFilterResult
    ) -> None:
        result.update_context(context)
