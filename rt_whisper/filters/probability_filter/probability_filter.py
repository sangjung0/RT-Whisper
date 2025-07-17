from __future__ import annotations
from typing import TYPE_CHECKING

from logging import Logger

from rt_whisper.abstracts import Worker
from rt_whisper.filters.probability_filter.service import (
    filter_probability_by_min_prob,
    filter_tokens_by_probability_outliers,
    update_statistics,
)
from rt_whisper.filters.probability_filter.data import (
    ProbabilityFilterParam,
    ProbabilityFilterResult,
    ProbabilityFilterState,
)

if TYPE_CHECKING:
    from rt_whisper.data import TokenState


class ProbabilityFilter(Worker):
    def __init__(self, z_thresh: float, min_prob: float, logger: Logger):
        super().__init__()
        self.logger = logger

        self.__Z_THRESH = z_thresh
        self.__MIN_PROB = min_prob

    # override
    def _register_state(self, context: TokenState) -> None:
        context.set_state(ProbabilityFilterState, ProbabilityFilterState())

    # override
    def _can_process(self, context: TokenState) -> ProbabilityFilterParam:
        if context.chunk.shape[0] > 0 and len(context.segment_tokens) > 0:
            prob_state = context.get_state(ProbabilityFilterState)
            return ProbabilityFilterParam.from_context(context, prob_state)
        return None

    # override
    def _process(self, param: ProbabilityFilterParam) -> ProbabilityFilterResult:
        self.logger.debug(f"Processing Probability Filter", group_level=1)

        self.logger.debug(f"Filtering tokens by minimum probability", group_level=2)
        self.logger.debug(
            f"Before: {''.join(str(t) for t in param.segment_tokens if t.is_word)}",
            group_level=2,
        )
        tokens = filter_probability_by_min_prob(
            param.segment_tokens, self.__MIN_PROB[param.language]
        )
        self.logger.debug(
            f"After: {''.join(str(t) for t in tokens if t.is_word)}", group_level=2
        )

        X = [t.probability for t in tokens if t.is_word]
        mean, std, N = update_statistics(X, param.mean, param.std, param.count)

        self.logger.debug(f"Filtering tokens by probability outliers", group_level=2)
        new_tokens = filter_tokens_by_probability_outliers(
            tokens, mean, std, self.__Z_THRESH[param.language]
        )
        self.logger.debug(
            f"After: {''.join(str(t) for t in new_tokens if t.is_word)}", group_level=2
        )

        return ProbabilityFilterResult(
            segment_tokens=new_tokens, mean=mean, std=std, count=N
        )

    # override
    def _update(self, state: TokenState, result: ProbabilityFilterResult) -> None:
        prob_state = state.get_state(ProbabilityFilterState)
        result.update_context(state, prob_state)
