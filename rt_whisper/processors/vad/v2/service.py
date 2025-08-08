from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rt_whisper.data import Token


def slice_vad_chunk(
    vad_chunk: np.ndarray,
    slice_point: int,
    timestamps_mapping: list[dict[str, int]],
):
    adjusted_slice_point = 0
    for ts in timestamps_mapping:
        offset = ts["offset"]
        if slice_point < ts["start"] + offset:
            break
        elif ts["end"] + offset < slice_point:
            adjusted_slice_point = ts["end"] + offset
            continue
        else:
            adjusted_slice_point = slice_point - offset
            break

    return vad_chunk[:adjusted_slice_point], vad_chunk[adjusted_slice_point:]


def add_offset(segment_tokens: list[Token], offset: int) -> None:
    for token in segment_tokens:
        token.start += offset
        token.end += offset


__all__ = [
    "slice_vad_chunk",
    "add_offset",
]
