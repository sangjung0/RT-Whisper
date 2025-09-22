from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rt_whisper.data import Token
    from rt_whisper.models.boundary_word_filter import BoundaryWordFilterWrapper


def filter_by_position_weighted(
    tokens: list[Token],
    offset: int,
    chunk_length: int,
    model: BoundaryWordFilterWrapper,
    non_apply_width: int = 0,
):
    start_list, end_list = [], []
    for token in tokens:
        if not token.is_word or token.start - offset < non_apply_width:
            continue

        start = token.start - offset
        end = token.end - offset
        start_list.append(start)
        end_list.append(end)

    weights = model(
        start=np.array(start_list), end=np.array(end_list), length=chunk_length
    )

    idx = 0
    for token in tokens:
        if not token.is_word or token.start - offset < non_apply_width:
            continue

        token.probability = weights[idx] * token.probability
        idx += 1

    return tokens


__all__ = ["filter_by_position_weighted"]
