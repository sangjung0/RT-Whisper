from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rt_whisper.data import Token
    from rt_whisper.models.boundary_word_filter import BoundaryWordFilter


def filter_by_position_weighted(
    tokens: list[Token],
    offset: int,
    chunk_length: int,
    head_boundary: float,
    tail_boundary: float,
    head_model: BoundaryWordFilter,
    tail_model: BoundaryWordFilter,
):
    for token in tokens:
        if not token.is_word:
            continue

        start = token.start - offset
        end = token.end - offset
        mid = (start + end) / 2
        dur = end - start

        if end < head_boundary:
            start, end, mid, dur = (
                start / head_boundary,
                end / head_boundary,
                mid / head_boundary,
                dur / head_boundary,
            )
            with torch.no_grad():
                weight = head_model(
                    torch.tensor([start, end, mid, dur], dtype=torch.float32)
                ).item()
            token.probability = token.probability * weight
        elif chunk_length - start < tail_boundary:
            start, end, mid = (
                chunk_length - end,
                chunk_length - start,
                chunk_length - mid,
            )
            start, end, mid, dur = (
                start / tail_boundary,
                end / tail_boundary,
                mid / tail_boundary,
                dur / tail_boundary,
            )
            with torch.no_grad():
                weight = tail_model(
                    torch.tensor([start, end, mid, dur], dtype=torch.float32)
                ).item()
            token.probability = token.probability * weight
        else:
            continue

    return tokens


__all__ = ["filter_by_position_weighted"]

