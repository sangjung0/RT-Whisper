from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from sj_utils.audio import generate_empty_chunk

if TYPE_CHECKING:
    from typing import Callable

    from rt_whisper.data import Token


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

    raise ValueError(
        f"Condition not found: \n\tc_index: {c_index} \n\tconditions: {conditions} \n\ttimestamp: {timestamp}"
    )

__all__ = [
    "vad",
    "merge_audio_from_timestamps",
    "generate_vad_timestamps_mapping",
    "set_offset",
]
