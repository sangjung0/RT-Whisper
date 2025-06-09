from dataclasses import dataclass
import numpy as np

from .sentence import Sentence
from .token import Token


@dataclass(slots=True, frozen=True)
class Result:
    completed: list[Sentence]
    candidate: list[Sentence]

    order: int
    offset: int
    recycle_chunk: np.ndarray
    recycle_segment_tokens: list[Token]

    recycles: dict[str, object]
