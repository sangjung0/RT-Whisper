from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from sj_utils.audio_utils import generate_empty_chunk

if TYPE_CHECKING:
    from rt_whisper.data import Token


def generate_vad_offset(prev_vad_offset: int, prev_vad_chunk: np.ndarray) -> int:
    """Generate the new VAD offset based on the previous VAD offset and chunk.

    Args:
        prev_vad_offset (int): Previous VAD offset.
        prev_vad_chunk (np.ndarray): Previous VAD chunk.

    Returns:
        int: New VAD offset calculated by adding the previous VAD offset
    """
    return prev_vad_offset + prev_vad_chunk.shape[0]


def apply_offset_to_timestamps(
    timestamps: list[dict[str, int]], offset: int
) -> list[dict[str, int]]:
    """Apply an offset to the start and end times of each segment in the timestamps.

    Args:
        timestamps (list[dict[str, int]]): List of dictionaries with 'start' and 'end' keys representing VAD timestamps.
        offset (int): Offset to apply to each timestamp.

    Returns:
        list[dict[str, int]]: List of dictionaries with 'start' and 'end' keys adjusted by the offset.
    """

    return (
        [
            {"start": segment["start"] + offset, "end": segment["end"] + offset}
            for segment in timestamps
        ]
        if timestamps
        else []
    )




def generate_context_vad_timestamps(
    context_vad_ts: list[dict[str, int]],
    vad_ts: list[dict[str, int]],
    anchor: int,
) -> list[dict[str, int]]:
    context_vad_ts = [ts for ts in (context_vad_ts + vad_ts) if ts["end"] > anchor]
    if context_vad_ts and context_vad_ts[0]["start"] < anchor:
        context_vad_ts[0]["start"] = anchor

    return context_vad_ts


def generate_context_vad_timestamps_mapping(
    merged_vad: list[dict[str, int]],
    anchor: int,
) -> list[dict[str, int]]:
    context_vad_mp = [ts for ts in merged_vad if ts["end"] + ts["offset"] > anchor]

    if (
        context_vad_mp
        and context_vad_mp[0]["start"] + context_vad_mp[0]["offset"] < anchor
    ):
        context_vad_mp[0]["start"] = anchor - context_vad_mp[0]["offset"]

    return context_vad_mp


def generate_context_vad_chunk(
    merged_vad_timestamps_mapping: list[dict[str, int]],
    merged_vad_chunk: np.ndarray,
) -> np.ndarray:
    if merged_vad_timestamps_mapping:
        anchor = merged_vad_chunk.shape[0] - (
            merged_vad_timestamps_mapping[-1]["end"]
            - merged_vad_timestamps_mapping[0]["start"]
        )
        return merged_vad_chunk[anchor:]
    else:
        return generate_empty_chunk()


def generate_context_vad_offset(
    vad_offset: int, vad_chunk: np.ndarray, context_vad_chunk: np.ndarray
) -> int:
    return vad_offset + vad_chunk.shape[0] - context_vad_chunk.shape[0]
