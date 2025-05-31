from typing import Union
import numpy as np
from pydantic import BaseModel, Field, field_validator

from .token import Token
from .sentence import Sentence
from .result import Result

STATISTIC = {
    "probability": {
        "mean": {},
        "std": {},
        "count": {},
    },
    "duration": {
        "mean": {},
        "std": {},
        "count": {},
    },
}


class Param(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
    }

    order: int = Field(0)
    sc_offset: int = Field(0)
    statistics: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=lambda: STATISTIC
    )
    prompt: Union[str, None] = Field(None)
    language: Union[str, None] = Field(None)

    audio: np.ndarray = Field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32), exclude=True
    )

    prev_audio: np.ndarray = Field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32), exclude=True
    )
    prev_processed_audio: np.ndarray = Field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32), exclude=True
    )
    prev_timestamps: list[dict[str, int]] = Field(default_factory=list)
    prev_timestamps_mapping: list[dict[str, int]] = Field(default_factory=list)
    prev_completed_tokens: list[Token] = Field(default_factory=list)
    prev_candidate_tokens: list[Token] = Field(default_factory=list)
    prev_candidate_sentences: list[Sentence] = Field(default_factory=list)
    prev_sentence: Union[Sentence, None] = Field(None)

    @classmethod
    @field_validator("audio", "prev_audio", "prev_processed_audio")
    def validate_audio(cls, v: np.ndarray):
        if isinstance(np.ndarray, v) and v.ndim == 1 and v.dtype == np.float32:
            return v
        raise ValueError("audio must be a 1D numpy array of float32")

    @staticmethod
    def reset_all_fields_to_default(obj: "Param"):
        for name, field in obj.__class__.model_fields.items():
            default = field.get_default(call_default_factory=True)
            setattr(obj, name, default)

    def update(self, result: Result):
        Param.reset_all_fields_to_default(self)

        self.order = result.order
        self.sc_offset = result.sc_offset
        self.statistics = result.statistics
        self.prev_audio = result.prev_audio
        self.prev_processed_audio = result.prev_processed_audio
        self.prev_timestamps = result.prev_timestamps
        self.prev_timestamps_mapping = result.prev_timestamps_mapping
        self.prev_completed_tokens = result.prev_completed_tokens
        self.prev_candidate_tokens = result.prev_candidate_tokens
        self.prev_candidate_sentences = result.candidate
        self.prev_sentence = result.prev_sentence
