from dataclasses import dataclass
from typing import Iterator
import numpy as np

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class ASRResult:
    merged_vad_chunk: np.ndarray
    candidate_tokens: Iterator[Token]
    language: str | None

    def update_context(self, context: Context) -> None:
        context.merged_vad_chunk = self.merged_vad_chunk
        context.candidate_tokens = self.candidate_tokens
        context.language = self.language
