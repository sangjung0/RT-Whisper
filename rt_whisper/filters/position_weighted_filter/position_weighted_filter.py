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
from rt_whisper.processors.asr.data import ASRState

if TYPE_CHECKING:
    from rt_whisper.rt_whisper_logger import RTWhisperLogger
    from rt_whisper.models.boundary_word_filter import BoundaryWordFilterWrapper
    from rt_whisper.data import TokenState


class PositionWeightedFilter(Worker):
    def __init__(
        self,
        model: BoundaryWordFilterWrapper,
        logger: RTWhisperLogger,
    ):
        super().__init__()
        self.logger = logger
        self.model = model

    # override
    def _can_process(self, context: TokenState):
        if context.chunk.shape[0] > 0:
            asr_state = context.get_state(ASRState)
            return PositionWeightedFilterParam.from_context(context, asr_state)
        return None

    # override
    def _process(
        self, param: PositionWeightedFilterParam
    ) -> PositionWeightedFilterResult:
        self.logger.debug(f"Processing Position Weighted Filter", group_level=1)

        self.logger.debug(
            f"Before: {''.join(str(t) for t in param.segment_tokens if t.is_word)}",
            group_level=2,
        )

        # TODO 여기서 offset은 이전 ASR에서 한번 업데이트되기 때문에, 현재 청크 이후의 offset이 됨. 따라서, 현재 청크의 길이를 빼줘야함. 설계 오류
        # TODO 여기서 chunk 길이와 이전 청크의 길이까지 빼야함.

        offset = param.offset - param.merged_chunk.shape[0]
        tokens = filter_by_position_weighted(
            param.segment_tokens,
            offset,
            param.merged_chunk.shape[0],
            self.model,
            self.model.boundary if offset == 0 else 0,
        )
        self.logger.debug(
            f"After: {''.join(str(t) for t in tokens if t.is_word)}", group_level=2
        )

        return PositionWeightedFilterResult(
            segment_tokens=tokens,
        )

    # override
    def _update(
        self, context: TokenState, result: PositionWeightedFilterResult
    ) -> None:
        result.update_context(context)


__all__ = ["PositionWeightedFilter"]
