from __future__ import annotations
from typing import TYPE_CHECKING

from rt_whisper.abstracts import Worker

from .data import PositionWeightedFilterParam, PositionWeightedFilterResult

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext


class PositionWeightedFilter(Worker):
    def __init__(
        self,
        boundary: float,
    ):
        super().__init__()
        self.__BOUNDARY = boundary

    # override
    def _can_process(self, context: TokenContext):
        if context.chunk.shape[0] > 0:
            return PositionWeightedFilterParam.from_context(context)
        return None

    # override
    def _process(
        self, param: PositionWeightedFilterParam
    ) -> PositionWeightedFilterResult:
        for token in param.segment_tokens:
            if not token.is_word:
                continue

            token.probability = self.__get_weighted_probability(
                token.probability,
                token.start - param.offset,
                token.end - param.offset,
                param.chunk.shape[0],
                self.__BOUNDARY,
            )

        return PositionWeightedFilterResult(
            # tokens=param.merged_candidate_tokens
        )

    # override
    def _update(
        self, context: TokenContext, result: PositionWeightedFilterResult
    ) -> None:
        result.update_context(context)

    def __get_weighted_probability(
        self,
        probabilities: float,
        start: int,
        end: int,
        duration: int,
        boundary: int,
    ) -> float:
        center = (start + end) / 2
        if center < duration - boundary:
            return probabilities
        return probabilities * ((duration - center) / boundary)
