from dataclasses import dataclass

import numpy as np

from rt_whisper.data import Context, Sentence, Token


@dataclass(slots=True)
class SentenceClassifierParam:
    sc_offset: int
    completed: list[Sentence]
    audio: np.ndarray
    audio_sc: int
    prev_audio: np.ndarray
    prev_audio_sc: int
    merged_processed_audio: np.ndarray
    merged_timestamps: list[dict[str, int]]
    merged_timestamps_mapping: list[dict[str, int]]
    prev_sentence: Sentence | None
    merged_candidate_tokens: list[Token]

    @staticmethod
    def from_context(context: Context) -> "SentenceClassifierParam":
        return SentenceClassifierParam(
            sc_offset=context.offset,
            completed=context.completed,
            audio=context.chunk,
            audio_sc=context.audio_sc,
            prev_audio=context.prev_chunk,
            prev_audio_sc=context.prev_chunk_size,
            merged_processed_audio=context.merged_vad_chunk,
            merged_timestamps=context.merged_vad_timestamps,
            merged_timestamps_mapping=context.merged_timestamps_mapping,
            prev_sentence=context.prev_sentence,
            merged_candidate_tokens=context.candidate_tokens,
        )
