from __future__ import annotations
from typing import TYPE_CHECKING
import statistics

from rt_whisper.abstracts import Worker
from rt_whisper.util.utils import update_mean_std

from .data import ProbabilityFilterParam, ProbabilityFilterResult

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext


class ProbabilityFilter(Worker):
    def __init__(
        self,
        z_thresh: float,
        min_prob: float,
    ):
        super().__init__()
        self.__Z_THRESH = z_thresh
        self.__MIN_PROB = min_prob

    # override
    def _can_process(self, context: TokenContext) -> ProbabilityFilterParam:
        if context.chunk.shape[0] > 0 and len(context.segment_tokens) > 0:
            return ProbabilityFilterParam.from_context(context)
        return None

    # override
    def _process(self, param: ProbabilityFilterParam) -> ProbabilityFilterResult:
        tokens = param.segment_tokens
        language = param.language
        prev_mean = param.mean
        prev_std = param.std
        prev_n = param.count

        tokens = [t for t in tokens if t.probability > self.__MIN_PROB[language]]
        X = [t.probability for t in tokens if t.is_word]

        if not X:
            return ProbabilityFilterResult(
                segment_tokens=tokens, mean=prev_mean, std=prev_std, count=prev_n
            )

        N = len(X)
        mean = statistics.mean(X)
        std = statistics.stdev(X) if len(X) > 1 else 0.0
        if prev_mean is not None and prev_std is not None and prev_n is not None:
            mean, std = update_mean_std(prev_mean, prev_std, prev_n, mean, std, N)

        new_tokens = []
        for t in tokens:
            if (
                t.is_word
                and t.probability < mean
                and mean - t.probability > self.__Z_THRESH[language] * std
            ):
                continue
            new_tokens.append(t)

        n = N + prev_n if prev_n is not None else N
        return ProbabilityFilterResult(
            segment_tokens=new_tokens, mean=mean, std=std, count=n
        )

    # override
    def _update(self, context: TokenContext, result: ProbabilityFilterResult) -> None:
        result.update_context(context)
