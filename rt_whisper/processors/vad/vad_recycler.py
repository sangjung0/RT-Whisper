from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from rt_whisper.util.utils import get_empty_chunk

from .vad_post_processor import VADPostProcessor
from .data import VADRecycleParam, VADRecycleResult

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext


class VADRecycler(VADPostProcessor):
    # override
    def _need_recycle(self, context: TokenContext) -> VADRecycleParam:
        return VADRecycleParam.from_context(context)

    # override
    def _recycle(self, param: VADRecycleParam):
        recycle_vad_timestamps = self.__get_recycle_vad_timestamps(
            prev_vad=param.prev_vad_timestamps,
            vad=param.vad_timestamps,
            anchor=param.anchor_timestamp,
        )

        recycle_vad_timestamps_mapping = self.__get_recycle_Vad_timestamps_mapping(
            merged_vad=param.merged_vad_timestamps_mapping,
            anchor=param.anchor_timestamp,
        )

        recycle_vad_chunk = self.__get_recycle_vad_chunk(
            merged_vad_timestamps_mapping=param.merged_vad_timestamps_mapping,
            vad_chunk=param.vad_chunk,
        )

        recycle_vad_offset = (
            param.vad_offset + param.vad_chunk.shape[0] - recycle_vad_chunk.shape[0]
        )

        return VADRecycleResult(
            recycle_vad_offset=recycle_vad_offset,
            recycle_vad_chunk=recycle_vad_chunk,
            recycle_vad_timestamps=recycle_vad_timestamps,
            recycle_vad_timestamps_mapping=recycle_vad_timestamps_mapping,
        )

    # override
    def _recycle_update(self, context: TokenContext, result: VADRecycleResult) -> None:
        result.update_context(context)

    def _get_prev_timestamps(self, timestamps: list[dict[str, int]], anchor: int):
        prev_timestamps = []
        t_index = 0
        while t_index < len(timestamps) and timestamps[t_index]["end"] < anchor:
            t_index += 1
        if t_index < len(timestamps) and timestamps[t_index]["start"] < anchor:
            t = timestamps[t_index]
            prev_timestamps.append({"start": anchor, "end": t["end"]})
            t_index += 1
        while t_index < len(timestamps):
            t = timestamps[t_index]
            prev_timestamps.append({"start": t["start"], "end": t["end"]})
            t_index += 1
        return prev_timestamps

    def _get_prev_timestamps_mapping(self, timestamps_mapping: list[dict], anchor: int):
        prev_timestamps_mapping = []
        t_index = 0
        while (
            t_index < len(timestamps_mapping)
            and timestamps_mapping[t_index]["end"] < anchor
        ):
            t_index += 1
        offset = (
            timestamps_mapping[t_index]["offset"]
            if t_index < len(timestamps_mapping)
            else anchor
        )
        if (
            t_index < len(timestamps_mapping)
            and timestamps_mapping[t_index]["start"] < anchor
        ):
            t = timestamps_mapping[t_index]
            prev_timestamps_mapping.append(
                {"start": 0, "end": t["end"] - anchor, "offset": 0}
            )
            t_index += 1
        while t_index < len(timestamps_mapping):
            t = timestamps_mapping[t_index]
            prev_timestamps_mapping.append(
                {
                    "start": t["start"] - anchor,
                    "end": t["end"] - anchor,
                    "offset": t["offset"] - offset,
                }
            )
            t_index += 1
        return prev_timestamps_mapping

    def __get_recycle_vad_timestamps(
        self,
        prev_vad: list[dict[str, int]],
        vad: list[dict[str, int]],
        anchor: int,
    ):

        recycle_vad_timestamps = [ts for ts in (prev_vad + vad) if ts["end"] > anchor]
        if recycle_vad_timestamps and recycle_vad_timestamps[0]["start"] < anchor:
            recycle_vad_timestamps[0]["start"] = anchor

        return recycle_vad_timestamps

    def __get_recycle_Vad_timestamps_mapping(
        self,
        merged_vad: list[dict[str, int]],
        anchor: int,
    ):
        recycle_vad = [ts for ts in merged_vad if ts["end"] + ts["offset"] > anchor]

        if recycle_vad and recycle_vad[0]["start"] + recycle_vad[0]["offset"] < anchor:
            recycle_vad[0]["start"] = anchor - recycle_vad[0]["offset"]

        return recycle_vad

    def __get_recycle_vad_chunk(
        self,
        merged_vad_timestamps_mapping: list[dict[str, int]],
        vad_chunk: np.ndarray,
    ):
        if merged_vad_timestamps_mapping:
            end = merged_vad_timestamps_mapping[-1]["end"]
            start = merged_vad_timestamps_mapping[0]["start"]
            anchor = vad_chunk.shape[0] - (end - start)
            # TODO 이거 vad_chunk가 아니라 merged_vad_chunk로 해야한다.
            return vad_chunk[anchor:]
        else:
            return get_empty_chunk()
