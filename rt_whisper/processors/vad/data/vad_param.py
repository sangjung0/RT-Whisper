from dataclasses import dataclass
import numpy as np

from rt_whisper.data import Context


@dataclass(slots=True)
class VadParam:
    chunk: np.ndarray
    offset: int

    @staticmethod
    def validate(context: Context) -> bool:
        return context.chunk_size > 0

    @staticmethod
    def from_context(context: Context):
        if VadParam.validate(context):
            return VadParam(
                chunk=context.chunk,
                offset=context.offset,
            )
        return None
