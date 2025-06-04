from rt_whisper.abstracts import Worker
from rt_whisper.data import Context

from .data import PositionWeightedFilterParam, PositionWeightedFilterResult


class PositionWeightedFilter(Worker):
    def __init__(
        self,
        boundary: float,
    ):
        super().__init__()
        self.__BOUNDARY = boundary

    # override
    def _can_process(self, context: Context):
        return PositionWeightedFilterParam.from_context(context)

    # override
    def _process(
        self, param: PositionWeightedFilterParam
    ) -> PositionWeightedFilterResult:
        merged_chunk_size = param.chunk_size + param.prev_chunk_size

        for token in param.candidate_tokens:
            if not token.is_word:
                continue

            token.probability = self.__get_weighted_probability(
                token.probability,
                token.start - param.prev_chunk_offset,
                token.end - param.prev_chunk_offset,
                merged_chunk_size,
                self.__BOUNDARY,
            )

        return PositionWeightedFilterResult(
            # tokens=param.merged_candidate_tokens
        )

    # override
    def _update(self, context: Context, result: PositionWeightedFilterResult) -> None:
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
