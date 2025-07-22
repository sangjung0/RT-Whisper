from __future__ import annotations
from typing import TYPE_CHECKING

import statistics

from sj_utils.statistics import update_mean_std

if TYPE_CHECKING:
    from rt_whisper.data import Token


def calculate_length(tokens: list[Token]) -> list[int]:
    return [t.end - t.start for t in tokens]


def calculate_length_ratio(tokens: list[Token], lengths: list[int]) -> list[float]:
    return [l / len(t.text) for t, l in zip(tokens, lengths)]


def filter_duration_by_min_dur(
    tokens: list[Token], length: list[float], min_dur: int
) -> tuple[list[Token], list[float]]:
    new_tokens = []
    new_length = []
    for t, l in zip(tokens, length):
        if t.is_word and l < min_dur:
            continue
        new_tokens.append(t)
        new_length.append(l)
    return new_tokens, new_length


def update_statistics(
    ratios: list[float], prev_mean: float, prev_std: float, prev_n: int
) -> tuple[float, float, int]:
    if not ratios:
        return prev_mean, prev_std, prev_n

    N = len(ratios)
    mean = statistics.mean(ratios)
    std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
    if prev_mean is not None and prev_std is not None and prev_n is not None:
        mean, std = update_mean_std(prev_mean, prev_std, prev_n, mean, std, N)
        N += prev_n

    return mean, std, N


def filter_tokens_by_duration_outliers(
    tokens: list[Token], ratios: list[float], mean: float, std: float, z_thresh: float
) -> list[Token]:
    return [
        t
        for t, r in zip(tokens, ratios)
        if not (t.is_word and r < mean and mean - r > z_thresh * std)
    ]


__all__ = [
    "calculate_length",
    "calculate_length_ratio",
    "filter_duration_by_min_dur",
    "update_statistics",
    "filter_tokens_by_duration_outliers",
]
