from __future__ import annotations
from typing import TYPE_CHECKING

import statistics

from sj_utils.statistics_utils import update_mean_std

if TYPE_CHECKING:
    from rt_whisper.data import Token


def calculate_length_ratio(tokens: list[Token]) -> list[float]:
    return [(t.end - t.start) / len(t.text.strip()) for t in tokens if t.is_word]


def update_statistics(
    X: list[float], prev_mean: float, prev_std: float, prev_n: int
) -> tuple[float, float, int]:
    if not X:
        return prev_mean, prev_std, prev_n

    adjusted_X = [x for x in X if x > 0]
    N = len(adjusted_X)
    mean = statistics.mean(adjusted_X)
    std = statistics.stdev(adjusted_X) if len(adjusted_X) > 1 else 0.0
    if prev_mean is not None and prev_std is not None and prev_n is not None:
        mean, std = update_mean_std(prev_mean, prev_std, prev_n, mean, std, N)
        N += prev_n

    return mean, std, N


def filter_tokens_by_duration_outliers(
    tokens: list[Token], X: list[float], mean: float, std: float, z_thresh: float
) -> list[Token]:
    new_tokens = []
    X_iter = iter(X)
    for t in tokens:
        if t.is_word:
            x = next(X_iter)
            if x < mean and mean - x > z_thresh * std:
                continue
        new_tokens.append(t)
    return new_tokens
