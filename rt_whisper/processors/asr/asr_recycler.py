from __future__ import annotations
from typing import TYPE_CHECKING

from .asr_processor import ASRProcessor
from .data import ASRRecycleParam, ASRRecycleResult

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext


class ASRRecycler(ASRProcessor):
    def __init__(
        self,
        *args,
        max_overlap_duration: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__MAX_OVERLAP_DURATION = max_overlap_duration

    # override
    def _need_recycle(self, context: TokenContext):
        return ASRRecycleParam.from_context(context)

    # override
    def _recycle(self, param: ASRRecycleParam) -> ASRRecycleResult:
        anchor = max(0, param.merged_chunk.shape[0] - self.__MAX_OVERLAP_DURATION)
        recycle_chunk = param.merged_chunk[anchor:]
        recycle_offset = param.offset + param.chunk.shape[0]
        anchor_timestamp = anchor + param.offset - param.prev_chunk.shape[0]

        return ASRRecycleResult(
            recycle_chunk=recycle_chunk,
            recycle_offset=recycle_offset,
            anchor_timestamp=anchor_timestamp,
        )

    # override
    def _recycle_update(self, context: TokenContext, result: ASRRecycleResult) -> None:
        result.update_context(context)
