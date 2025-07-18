from __future__ import annotations
from typing import TYPE_CHECKING

import statistics

from sj_utils.statistics_utils import update_mean_std

if TYPE_CHECKING:
    from rt_whisper.data import Token


def calculate_length_ratio(tokens: list[Token]) -> list[float]:
    return [(t.end - t.start) / len(t.text.strip()) for t in tokens if t.is_word]


def filter_duration_by_min_dur(
    tokens: list[Token], X: list[float], min_dur: int
) -> tuple[list[Token], list[float]]:
    new_tokens = []
    new_X = []
    for t, x in zip(tokens, X):
        if t.is_word and x < min_dur:
            continue
        new_tokens.append(t)
        new_X.append(x)
    return new_tokens, new_X


def update_statistics(
    X: list[float], prev_mean: float, prev_std: float, prev_n: int
) -> tuple[float, float, int]:
    if not X:
        return prev_mean, prev_std, prev_n

    N = len(X)
    mean = statistics.mean(X)
    std = statistics.stdev(X) if len(X) > 1 else 0.0
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
