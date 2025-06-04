from dataclasses import dataclass
from typing import Union
import numpy as np

from rt_whisper.data import Context


@dataclass(slots=True)
class ASRParam:
    vad_chunk: np.ndarray
    prev_vad_chunk: np.ndarray
    prev_chunk_offset: int
    language: Union[str, None]
    prompt: Union[str, None]

    @staticmethod
    def validate(context: Context) -> bool:
        return context.vad_chunk_size > 0

    @staticmethod
    def from_context(context: Context) -> "ASRParam":
        if not ASRParam.validate(context):
            return None
        return ASRParam(
            vad_chunk=context.vad_chunk,
            prev_vad_chunk=context.prev_vad_chunk,
            prev_chunk_offset=context.prev_chunk_offset,
            language=context.language,
            prompt=context.prompt,
        )
