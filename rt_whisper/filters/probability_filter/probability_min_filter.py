from __future__ import annotations
from typing import TYPE_CHECKING

from rt_whisper.abstracts import Worker
from rt_whisper.filters.probability_filter.data import (
    ProbabilityMinFilterParam,
    ProbabilityMinFilterResult,
)
from rt_whisper.filters.probability_filter.service import filter_probability_by_min_prob

if TYPE_CHECKING:
    from rt_whisper import RTWhisperLogger
    from rt_whisper.data import TokenState


class ProbabilityMinFilter(Worker):
    def __init__(self, min_prob: dict[str, float], logger: RTWhisperLogger):
        super().__init__()
        self.logger = logger

        self.__MIN_PROB = min_prob

    # override
    def _can_process(self, state: TokenState) -> ProbabilityMinFilterParam | None:
        if len(state.segment_tokens) > 0:
            return ProbabilityMinFilterParam.from_context(state)
        return None

    # override
    def _process(self, param: ProbabilityMinFilterParam) -> ProbabilityMinFilterResult:
        self.logger.debug(f"Processing Probability Min Filter", group_level=1)

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

        return ProbabilityMinFilterResult(
            segment_tokens=tokens,
        )

    # override
    def _update(self, state: TokenState, result: ProbabilityMinFilterResult) -> None:
        result.update_context(state)
