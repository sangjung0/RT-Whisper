from dataclasses import dataclass

import numpy as np

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class TokenClassifierParam:
    offset: int
    chunk: np.ndarray
    chunk_size: int
    vad_timestamps: list[dict[str, int]]
    prev_chunk: np.ndarray
    prev_chunk_offset: int
    prev_chunk_size: int
    prev_completed_tokens: list[Token]
    prev_vad_timestamps: list[dict[str, int]]
    candidate_tokens: list[Token]
    merged_vad_chunk: np.ndarray
    merged_vad_chunk_size: int
    merged_vad_timestamps_mapping: list[dict[str, int]]

    @staticmethod
    def from_context(context: Context) -> "TokenClassifierParam":
        return TokenClassifierParam(
            offset=context.offset,
            chunk=context.chunk,
            chunk_size=context.chunk_size,
            vad_timestamps=context.vad_timestamps,
            prev_chunk=context.prev_chunk,
            prev_chunk_size=context.prev_chunk_size,
            prev_chunk_offset=context.prev_chunk_offset,
            prev_completed_tokens=context.prev_completed_tokens,
            prev_vad_timestamps=context.prev_vad_timestamps,
            candidate_tokens=context.candidate_tokens,
            merged_vad_chunk=context.merged_vad_chunk,
            merged_vad_chunk_size=context.merged_vad_chunk_size,
            merged_vad_timestamps_mapping=context.merged_vad_timestamps_mapping,
        )
