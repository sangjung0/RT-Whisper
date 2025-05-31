from dataclasses import dataclass
from typing import Union
import numpy as np
from beartype import beartype

from .sentence_ import Sentence
from .token_ import Token


@beartype
@dataclass(frozen=True)
class Result:
    order: int
    sc_offset: int
    statistics: dict[str, dict[str, dict[str, float]]]
    completed: list[Sentence]
    candidate: list[Sentence]
    audio: np.ndarray
    processed_audio: np.ndarray
    prev_audio: np.ndarray
    prev_processed_audio: np.ndarray
    prev_timestamps: list[dict]
    prev_timestamps_mapping: list[dict]
    prev_completed_tokens: list[Token]
    prev_candidate_tokens: list[Token]
    prev_sentence: Union[Sentence, None]
