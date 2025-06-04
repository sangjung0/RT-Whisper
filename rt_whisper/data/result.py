from dataclasses import dataclass
from typing import Union
import numpy as np
from beartype import beartype

from .sentence import Sentence
from .token import Token


@beartype
@dataclass(frozen=True)
class Result:
    order: int
    next_offset: int
    statistics: dict[str, dict[str, dict[str, float | None]]]
    completed: list[Sentence]
    candidate: list[Sentence]
    chunk: np.ndarray
    vad_chunk: np.ndarray
    recycle_chunk: np.ndarray
    recycle_vad_chunk: np.ndarray
    recycle_vad_timestamps: list[dict]
    recycle_vad_timestamps_mapping: list[dict]
    recycle_completed_tokens: list[Token]
    recycle_candidate_tokens: list[Token]
    prev_sentence: Union[Sentence, None]
