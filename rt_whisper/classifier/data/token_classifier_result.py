from dataclasses import dataclass

import numpy as np

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class TokenClassifierResult:
    next_offset: int
    recycle_chunk: np.ndarray
    recycle_vad_chunk: np.ndarray
    recycle_vad_timestamps: list[dict[str, int]]
    recycle_vad_timestamps_mapping: list[dict[str, int]]
    recycle_candidate_tokens: list[Token]
    completed_tokens: list[Token]

    def update_context(self, context: Context) -> None:
        context.next_offset = self.next_offset
        context.recycle_chunk = self.recycle_chunk
        context.recycle_vad_chunk = self.recycle_vad_chunk
        context.recycle_vad_timestamps = self.recycle_vad_timestamps
        context.recycle_vad_timestamps_mapping = self.recycle_vad_timestamps_mapping
        context.recycle_candidate_tokens = self.recycle_candidate_tokens
        context.completed_tokens = self.completed_tokens
