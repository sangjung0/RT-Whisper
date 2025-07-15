from __future__ import annotations
from typing import TYPE_CHECKING

from logging import Logger

from rt_whisper.abstracts import Worker
from rt_whisper.filters.duration_filter.service import (
    filter_tokens_by_duration_outliers,
    update_statistics,
)
from rt_whisper.filters.duration_filter.data import (
    DurationFilterParam,
    DurationFilterResult,
    DurationFilterState,
)
from rt_whisper.filters.duration_filter.service import calculate_length_ratio

if TYPE_CHECKING:
    from rt_whisper.data import TokenState


class DurationFilter(Worker):
    def __init__(self, z_thresh: dict[str:float], logger: Logger):
        super().__init__()
        self.logger = logger
        self.__Z_THRESH = z_thresh

    # override
    def _register_state(self, state: TokenState):
        state.set_state(DurationFilterState, DurationFilterState())

    # override
    def _can_process(self, state: TokenState) -> DurationFilterParam:
        if state.chunk.shape[0] > 0 and len(state.segment_tokens) > 0:
            dfs_state = state.get_state(DurationFilterState)
            return DurationFilterParam.from_state(state, dfs_state)
        return None

    # override
    def _process(self, param: DurationFilterParam) -> DurationFilterResult:
        self.logger.debug(f"\tProcessing Duration Filter")

        X = calculate_length_ratio(param.segment_tokens)
        mean, std, n = update_statistics(X, param.mean, param.std, param.count)

        self.logger.debug(f"\t\tBefore: {param.segment_tokens}")
        tokens = filter_tokens_by_duration_outliers(
            param.segment_tokens, X, mean, std, self.__Z_THRESH[param.language]
        )
        self.logger.debug(f"\t\tAfter: {tokens}")

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
