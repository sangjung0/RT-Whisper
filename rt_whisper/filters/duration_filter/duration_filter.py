import statistics

from rt_whisper.abstracts import Worker
from rt_whisper.data import Context
from rt_whisper.util.utils import update_mean_std

from .data import DurationFilterParam, DurationFilterResult


class DurationFilter(Worker):
    def __init__(self, z_thresh: dict[str:float]):
        super().__init__()
        self.__Z_THRESH = z_thresh

    # override
    def _can_process(self, context: Context) -> DurationFilterParam:
        return DurationFilterParam.from_context(context)

    # override
    def _process(self, param: DurationFilterParam) -> DurationFilterResult:
        candidate_tokens = param.candidate_tokens
        language = param.language
        prev_mean = param.mean
        prev_std = param.std
        prev_n = param.count

        X = [
            (t.end - t.start) / len(t.text.strip())
            for t in candidate_tokens
            if t.is_word
        ]

        if not X:
            return DurationFilterResult(
                candidate_tokens=candidate_tokens,
                mean=prev_mean,
                std=prev_std,
                count=prev_n,
            )

        adjusted_X = [x for x in X if x > 0]
        N = len(adjusted_X)
        mean = statistics.mean(adjusted_X)
        std = statistics.stdev(adjusted_X) if len(adjusted_X) > 1 else 0.0
        if prev_mean is not None and prev_std is not None and prev_n is not None:
            mean, std = update_mean_std(prev_mean, prev_std, prev_n, mean, std, N)

        new_candidate_tokens = []
        X_iter = iter(X)
        for t in candidate_tokens:
            if t.is_word:
                x = next(X_iter)
                if x < mean and mean - x > self.__Z_THRESH[language] * std:
                    continue
            new_candidate_tokens.append(t)

        n = N + prev_n if prev_n is not None else N

        return DurationFilterResult(
            candidate_tokens=new_candidate_tokens, mean=mean, std=std, count=n
        )

    # override
    def _update(self, context: Context, result: DurationFilterResult) -> None:
        result.update_context(context)
