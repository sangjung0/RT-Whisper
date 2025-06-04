from dataclasses import dataclass

import numpy as np

from rt_whisper.data import Context, Sentence


@dataclass(slots=True)
class SentenceClassifierResult:
    sc_offset: int
    prev_audio: np.ndarray
    prev_processed_audio: np.ndarray
    prev_timestamps: list[dict[str, int]]
    prev_timestamps_mapping: list[dict[str, int]]
    prev_sentence: Sentence | None

    def update_context(self, context: Context) -> None:
        context.offset = self.sc_offset
        context.prev_chunk = self.prev_audio
        context.prev_vad_chunk = self.prev_processed_audio
        context.prev_vad_timestamps = self.prev_timestamps
        context.prev_vad_timestamps_mapping = self.prev_timestamps_mapping
        context.prev_sentence = self.prev_sentence
