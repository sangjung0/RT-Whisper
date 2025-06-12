from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .vad_processor import VADProcessor
from .data import (
    VADPostParam,
    VADPostResult,
)

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext, Token


class VADPostProcessor(VADProcessor):
    # override
    def _can_post_process(self, context: TokenContext) -> VADPostParam:
        return VADPostParam.from_context(context)

    # override
    def _post_process(self, param: VADPostParam) -> VADPostResult:
        # 청크 합치기
        merged_chunk = np.concatenate([param.prev_chunk, param.chunk], axis=0)

        merged_vad_timestamps_mapping, vad_timestamps_mapping = (
            self.__get_merged_vad_timestamps_mapping(
                param.prev_vad_timestamps_mapping,
                param.vad_timestamps,
                param.vad_offset,
            )
        )

        self.__set_offset(
            param.segment_tokens,
            merged_vad_timestamps_mapping,
        )

        return VADPostResult(
            merged_chunk=merged_chunk,
            vad_timestamps_mapping=vad_timestamps_mapping,
            merged_vad_timestamps_mapping=merged_vad_timestamps_mapping,
        )

    # override
    def _post_update(self, context: TokenContext, result: VADPostResult):
        result.update_context(context)

    def __set_offset(
        self,
        segment_tokens: list[Token],
        merged_vad_timestamps_mapping: list[dict[str, int]],
    ):
        c_index = 0
        for token in segment_tokens:
            c_index, offset = self.__find_condition(
                c_index, merged_vad_timestamps_mapping, token.start
            )
            token.start = token.start + offset
            c_index, offset = self.__find_condition(
                c_index, merged_vad_timestamps_mapping, token.end
            )
            token.end = token.end + offset

    def __get_merged_vad_timestamps_mapping(
        self,
        prev_vad_timestamps_mapping: list[dict[str, int]],
        vad_timestamps: list[dict[str, int]],
        vad_offset: int,
    ):
        prev_end = (
            prev_vad_timestamps_mapping[-1]["end"]
            if prev_vad_timestamps_mapping
            else vad_offset
        )
        vad_timestamps_mapping = []
        for ts in vad_timestamps:
            duration = ts["end"] - ts["start"]
            end = prev_end + duration
            vad_timestamps_mapping.append(
                {
                    "start": prev_end,
                    "end": end,
                    "offset": ts["end"] - end,
                }
            )
            prev_end = end

        merged_vad_timestamps_mapping = (
            prev_vad_timestamps_mapping + vad_timestamps_mapping
        )

        return merged_vad_timestamps_mapping, vad_timestamps_mapping

    def __find_condition(
        self, c_index: int, conditions: list[dict[str, int]], timestamp: int
    ):
        while c_index < len(conditions) and c_index >= 0:
            c = conditions[c_index]
            start = c["start"]
            end = c["end"]
            offset = c["offset"]
            if start <= timestamp <= end:
                return c_index, offset
            elif timestamp < start:
                c_index -= 1
            else:
                c_index += 1

        print(f"c_index: {c_index}, conditions: {conditions}, timestamp: {timestamp}")
        raise ValueError("Condition not found")
