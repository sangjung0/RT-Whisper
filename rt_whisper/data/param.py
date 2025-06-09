from dataclasses import dataclass, field
import numpy as np

from .result import Result
from .token import Token

generate_empty_chunk = lambda: np.zeros((0,), dtype=np.float32)


@dataclass(slots=True)
class Param:
    chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    offset: int = field(default=0)
    order: int = field(default=0)
    prompt: str | None = field(default=None)
    language: str | None = field(default=None)

    recycle_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    recycle_segment_tokens: list[Token] = field(default_factory=list)

    recycles: dict = field(default_factory=dict)

    def validate_audio(self, v: np.ndarray):
        return isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype == np.float32

    def __post_init__(self):
        if not self.validate_audio(self.chunk) or not self.validate_audio(
            self.recycle_chunk
        ):
            raise ValueError("chunk must be a 1D numpy array of float32")

    def update(self, result: Result):
        self.__init__()

        self.offset = result.offset
        self.order = result.order

        self.recycle_chunk = result.recycle_chunk
        self.recycle_segment_tokens = result.recycle_segment_tokens
        self.recycles = result.recycles
