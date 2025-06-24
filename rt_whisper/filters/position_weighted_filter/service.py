from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rt_whisper.data import Token


def filter_by_position_weighted(
    tokens: list[Token], offset: int, chunk_length: int, boundary: float
):
    for token in tokens:
        if not token.is_word:
            continue

        token.probability = __get_weighted_probability(
            token.probability,
            token.start - offset,
            token.end - offset,
            chunk_length,
            boundary,
        )

    return tokens


def __get_weighted_probability(
    probability: float,
    start: int,
    end: int,
    duration: int,
    boundary: float,
):
    center = (start + end) / 2
    if center < duration - boundary:
        return probability
    else:
        return probability * ((duration - center) / boundary)
