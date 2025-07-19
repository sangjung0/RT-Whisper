from __future__ import annotations
from typing import TYPE_CHECKING

from rt_whisper.abstracts import Worker
from rt_whisper.filters.duration_filter.data import (
    DurationFilterParam,
    DurationFilterResult,
    DurationFilterState,
)
from rt_whisper.filters.duration_filter.service import (
    calculate_length,
    calculate_length_ratio,
    filter_duration_by_min_dur,
    filter_tokens_by_duration_outliers,
    update_statistics,
)

if TYPE_CHECKING:
    from rt_whisper import RTWhisperLogger
    from rt_whisper.data import TokenState


class DurationFilter(Worker):
    def __init__(
        self, z_thresh: dict[str:float], min_dur: int, logger: RTWhisperLogger
    ):
        super().__init__()
        self.logger = logger
        self.__Z_THRESH = z_thresh
        self.__MIN_DUR = min_dur

    # override
    def _register_state(self, state: TokenState):
        state.set_state(DurationFilterState, DurationFilterState())

    # override
    def _can_process(self, state: TokenState) -> DurationFilterParam:
        if len(state.segment_tokens) > 0:
            dfs_state = state.get_state(DurationFilterState)
            return DurationFilterParam.from_state(state, dfs_state)
        return None

    # override
    def _process(self, param: DurationFilterParam) -> DurationFilterResult:
        self.logger.debug(f"Processing Duration Filter", group_level=1)

        lengths = calculate_length(param.segment_tokens)

        self.logger.debug(f"Filtering tokens by minimum duration", group_level=2)
        self.logger.debug(
            f"Before: {''.join(str(t) for t in param.segment_tokens if t.is_word)}",
            group_level=2,
        )
        tokens, lengths = filter_duration_by_min_dur(
            param.segment_tokens, lengths, self.__MIN_DUR[param.language]
        )
        self.logger.debug(
            f"After: {''.join(str(t) for t in tokens if t.is_word)}", group_level=2
        )

        self.logger.debug(f"Filtering tokens by duration outliers", group_level=2)
        ratios = calculate_length_ratio(tokens, lengths)
        t_ratios = [r for t, r in zip(tokens, ratios) if t.is_word]
        mean, std, n = update_statistics(t_ratios, param.mean, param.std, param.count)
        tokens = filter_tokens_by_duration_outliers(
            tokens, ratios, mean, std, self.__Z_THRESH[param.language]
        )
        self.logger.debug(
            f"After: {''.join(str(t) for t in tokens if t.is_word)}", group_level=2
        )

        return DurationFilterResult(
            segment_tokens=tokens,
            mean=mean,
            std=std,
            count=n,
        )

    # override
    def _update(self, state: TokenState, result: DurationFilterResult) -> None:
        dfs_state = state.get_state(DurationFilterState)
        result.update_state(state, dfs_state)
