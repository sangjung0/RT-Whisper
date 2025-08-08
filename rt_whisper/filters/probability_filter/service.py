from __future__ import annotations
from typing import TYPE_CHECKING
import statistics

from sj_utils.statistics import update_mean_std

if TYPE_CHECKING:
    from rt_whisper.data import Token


def filter_probability_by_min_prob(tokens: list[Token], min_prob: float):
    return [t for t in tokens if t.probability > min_prob]


def update_statistics(X: list[float], prev_mean: float, prev_std: float, prev_n: int):
    if not X:
        return prev_mean, prev_std, prev_n

    N = len(X)
    mean = statistics.mean(X)
    std = statistics.stdev(X) if len(X) > 1 else 0.0
    if prev_mean is not None and prev_std is not None and prev_n is not None:
        mean, std = update_mean_std(prev_mean, prev_std, prev_n, mean, std, N)
        N += prev_n

    return mean, std, N


def filter_tokens_by_probability_outliers(
    tokens: list[Token], mean: float, std: float, z_thresh: float
) -> list[Token]:
    new_tokens = []
    for t in tokens:
        if t.is_word and t.probability < mean and mean - t.probability > z_thresh * std:
            continue
        new_tokens.append(t)
    return new_tokens


__all__ = [
    "filter_probability_by_min_prob",
    "update_statistics",
    "filter_tokens_by_probability_outliers",
]
