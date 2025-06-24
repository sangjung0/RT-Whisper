from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Callable
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


def vad(
    audio: np.ndarray, vader: Callable[[np.ndarray], list[dict[str, int]]]
) -> list[dict[str, int]]:
    """Get the VAD timestamps for the given audio using the provided VAD function.

    Args:
        audio (np.ndarray): Audio data to process.
        vader (Callable[[np.ndarray], list[dict[str, int]]]): Function to process audio and return VAD timestamps.

    Returns:
        list[dict[str, int]]: List of dictionaries with 'start' and 'end' keys representing VAD timestamps.
    """

    return [] if audio.shape[0] == 0 else vader(audio)


def merge_audio_from_timestamps(
    audio: np.ndarray, timestamps: list[dict[str, int]]
) -> np.ndarray:
    """Merge audio segments based on the provided timestamps.

    Args:
        audio (np.ndarray): Audio data to process.
        timestamps (list[dict[str, int]]): List of dictionaries with 'start' and 'end' keys representing VAD timestamps.

    Returns:
        np.ndarray: Merged audio segments as a single NumPy array.
    """

    if not timestamps:
        return generate_empty_chunk()

    return np.concatenate(
        [audio[segment["start"] : segment["end"]] for segment in timestamps]
    )


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


def generate_vad_timestamps_mapping(
    prev_end: int,
    vad_timestamps: list[dict[str, int]],
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    vad_timestamps_mapping = []
    for ts in vad_timestamps:
        end = prev_end + ts["end"] - ts["start"]
        vad_timestamps_mapping.append(
            {
                "start": prev_end,
                "end": end,
                "offset": ts["end"] - end,
            }
        )
        prev_end = end

    return vad_timestamps_mapping


def set_offset(
    segment_tokens: list[Token],
    merged_vad_timestamps_mapping: list[dict[str, int]],
) -> None:
    c_index = 0
    for token in segment_tokens:
        c_index, offset = __find_condition(
            c_index, merged_vad_timestamps_mapping, token.start
        )
        token.start = token.start + offset
        c_index, offset = __find_condition(
            c_index, merged_vad_timestamps_mapping, token.end
        )
        token.end = token.end + offset


def __find_condition(
    c_index: int, conditions: list[dict[str, int]], timestamp: int
) -> tuple[int, int]:
    while c_index < len(conditions) and c_index >= 0:
        condition = conditions[c_index]
        start = condition["start"]
        end = condition["end"]
        offset = condition["offset"]
        if start <= timestamp <= end:
            return c_index, offset
        elif timestamp < start:
            c_index -= 1
        else:
            c_index += 1

    print(f"c_index: {c_index}, conditions: {conditions}, timestamp: {timestamp}")
    raise ValueError("Condition not found")


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
