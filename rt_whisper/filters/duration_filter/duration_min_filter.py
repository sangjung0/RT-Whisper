from __future__ import annotations
from typing import TYPE_CHECKING

from rt_whisper.abstracts import Worker
from rt_whisper.filters.duration_filter.data import (
    DurationMinFilterParam,
    DurationMinFilterResult,
)
from rt_whisper.filters.duration_filter.service import (
    calculate_length,
    filter_duration_by_min_dur,
)

if TYPE_CHECKING:
    from rt_whisper import RTWhisperLogger
    from rt_whisper.data import TokenState


class DurationMinFilter(Worker):
    def __init__(self, min_dur: int, logger: RTWhisperLogger):
        super().__init__()
        self.logger = logger
        self.__MIN_DUR = min_dur

    # override
    def _can_process(self, state: TokenState) -> DurationMinFilterParam:
        if len(state.segment_tokens) > 0:
            return DurationMinFilterParam.from_state(state)
        return None

    # override
    def _process(self, param: DurationMinFilterParam) -> DurationMinFilterResult:
        self.logger.debug(f"Processing Duration Min Filter", group_level=1)

        lengths = calculate_length(param.segment_tokens)
        self.logger.debug(f"Filtering tokens by minimum duration", group_level=2)
        self.logger.debug(
            f"Before: {''.join(str(t) for t in param.segment_tokens if t.is_word)}",
            group_level=2,
        )
        tokens, _ = filter_duration_by_min_dur(
            param.segment_tokens, lengths, self.__MIN_DUR[param.language]
        )
        self.logger.debug(
            f"After: {''.join(str(t) for t in tokens if t.is_word)}", group_level=2
        )

        return DurationMinFilterResult(segment_tokens=tokens)

    # override
    def _update(self, state: TokenState, result: DurationMinFilterResult) -> None:
        result.update_state(state)
