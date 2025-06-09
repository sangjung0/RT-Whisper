from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Callable

import numpy as np
from rt_whisper.abstracts import Worker
from rt_whisper.util.utils import get_empty_chunk

from .data import (
    VADProcessParam,
    VADProcessResult,
)

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext


class VADProcessor(Worker):
    def __init__(self, vad: Callable[[np.ndarray], list[dict[str, int]]]):
        super().__init__()
        self.__vad = vad

    # override
    def _can_process(self, context: TokenContext) -> VADProcessParam:
        return VADProcessParam.from_context(context)

    # override
    def _process(self, param: VADProcessParam) -> None:
        vad_offset = param.prev_vad_offset + param.prev_vad_chunk.shape[0]
        timestamps = self.__get_timestamps(param.chunk)

        vad_chunk, vad_timestamps = self.__get_vad_chunk(
            audio=param.chunk,
            timestamps=timestamps,
            offset=param.offset,
        )

        return VADProcessResult(
            vad_chunk=vad_chunk,
            vad_offset=vad_offset,
            prev_vad_chunk=param.prev_vad_chunk,
            vad_timestamps=vad_timestamps,
        )

    # override
    def _update(self, context: TokenContext, result: VADProcessResult) -> None:
        result.update_context(context)

    def __get_timestamps(self, audio: np.ndarray):
        return [] if audio.shape[0] == 0 else self.__vad(audio)

    def __get_vad_chunk(
        self, audio: np.ndarray, timestamps: list[dict[str, int]], offset: int
    ) -> np.ndarray:
        if not timestamps:
            return get_empty_chunk(), []

        merged_audio = []
        vad_timestamps = []
        for segment in timestamps:
            start = segment["start"]
            end = segment["end"]
            merged_audio.append(audio[start:end])
            vad_timestamps.append({"start": start + offset, "end": end + offset})

        vad_chunk = np.concatenate(merged_audio)
        return vad_chunk, vad_timestamps
