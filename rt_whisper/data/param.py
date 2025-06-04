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

empty_chunk = lambda: np.zeros((0,), dtype=np.float32)


class Param(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
    }

    order: int = Field(default=0)
    offset: int = Field(default=0)
    statistics: dict[str, dict[str, dict[str, float | None]]] = Field(
        default_factory=lambda: STATISTIC
    )
    prompt: str | None = Field(default=None)
    language: str | None = Field(default=None)

    chunk: np.ndarray = Field(default_factory=empty_chunk, exclude=True)

    recycle_chunk: np.ndarray = Field(default_factory=empty_chunk, exclude=True)
    recycle_vad_chunk: np.ndarray = Field(default_factory=empty_chunk, exclude=True)
    recycle_vad_timestamps: list[dict[str, int]] = Field(default_factory=list)
    recycle_vad_timestamps_mapping: list[dict[str, int]] = Field(default_factory=list)
    recycle_completed_tokens: list[Token] = Field(default_factory=list)
    recycle_candidate_tokens: list[Token] = Field(default_factory=list)

    prev_candidate_sentences: list[Sentence] = Field(default_factory=list)
    prev_sentence: Sentence | None = Field(default=None)

    @classmethod
    @field_validator("chunk", "recycle_chunk", "recycle_vad_chunk")
    def validate_audio(cls, v: np.ndarray):
        if isinstance(np.ndarray, v) and v.ndim == 1 and v.dtype == np.float32:
            return v
        raise ValueError("chunk must be a 1D numpy array of float32")

    @staticmethod
    def reset_all_fields_to_default(obj: "Param"):
        for name, field in obj.__class__.model_fields.items():
            default = field.get_default(call_default_factory=True)
            setattr(obj, name, default)

    def update(self, result: Result):
        Param.reset_all_fields_to_default(self)

        self.order = result.order
        self.offset = result.next_offset
        self.statistics = result.statistics
        self.recycle_chunk = result.recycle_chunk
        self.recycle_vad_chunk = result.recycle_vad_chunk
        self.recycle_vad_timestamps = result.recycle_vad_timestamps
        self.recycle_vad_timestamps_mapping = result.recycle_vad_timestamps_mapping
        self.recycle_completed_tokens = result.recycle_completed_tokens
        self.recycle_candidate_tokens = result.recycle_candidate_tokens
        self.prev_candidate_sentences = result.candidate
        self.prev_sentence = result.prev_sentence
