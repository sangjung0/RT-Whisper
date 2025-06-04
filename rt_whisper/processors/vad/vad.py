from typing import Callable
import numpy as np

from rt_whisper.abstracts import Worker
from rt_whisper.data import Context

from .data import (
    VadParam,
    VadResult,
    VadPostParam,
    VadPostResult,
)


class VAD(Worker):

    def __init__(self, vad: Callable[[np.ndarray], list[dict[str, int]]]):
        super().__init__()
        self.__vad = vad

    # main process
    # override
    def _can_process(self, context: Context) -> VadParam:
        return VadParam.from_context(context)

    # override
    def _process(self, param: VadParam) -> VadResult:
        timestamps = self.__vad(param.chunk)

        merged_audio = []
        vad_timestamps = []
        for segment in timestamps:
            start = segment["start"]
            end = segment["end"]
            merged_audio.append(param.chunk[start:end])
            vad_timestamps.append(
                {"start": start + param.offset, "end": end + param.offset}
            )

        vad_chunk = (
            np.concatenate(merged_audio)
            if merged_audio
            else np.zeros((0,), dtype=np.float32)
        )

        return VadResult(vad_chunk=vad_chunk, vad_timestamps=vad_timestamps)

    # override
    def _update(self, context: Context, result: VadResult) -> None:
        result.update_context(context)

    # override
    def _can_post_process(self, context: Context) -> bool:
        return VadPostParam.from_context(context)

    # override
    def _post_process(self, param: VadPostParam) -> VadPostResult:

        vad_timestamps_mapping = []
        prev_end = (
            param.prev_vad_timestamps_mapping[-1]["end"]
            if param.prev_vad_timestamps_mapping
            else 0
        )
        for ts in param.vad_timestamps:
            duration = ts["end"] - ts["start"]
            end = prev_end + duration
            vad_timestamps_mapping.append(
                {"start": prev_end, "end": end, "offset": ts["end"] - end}
            )
            prev_end = end

        merged_vad_timestamps_mapping = (
            param.prev_vad_timestamps_mapping + vad_timestamps_mapping
        )

        c_index = 0
        for token in param.candidate_tokens:
            c_index, offset = self.__find_condition(
                c_index, merged_vad_timestamps_mapping, token.start
            )
            token.start = token.start + offset
            c_index, offset = self.__find_condition(
                c_index, merged_vad_timestamps_mapping, token.end
            )
            token.end = token.end + offset

        return VadPostResult(
            vad_timestamps_mapping=vad_timestamps_mapping,
            merged_vad_timestamps_mapping=merged_vad_timestamps_mapping,
            # tokens=param.candidate_tokens
        )

    # override
    def _post_update(self, context: Context, result: VadPostResult):
        result.update_context(context)

    def __find_condition(
        self, c_index: int, conditions: list[dict[str, int]], sample_count: int
    ):
        while c_index < len(conditions) or c_index >= 0:
            c = conditions[c_index]
            start = c["start"]
            end = c["end"]
            offset = c["offset"]
            if start <= sample_count <= end:
                return c_index, offset
            elif sample_count < start:
                c_index -= 1
            else:
                c_index += 1

        raise ValueError("Condition not found")
