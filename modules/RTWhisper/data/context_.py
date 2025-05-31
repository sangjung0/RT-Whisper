from dataclasses import MISSING, dataclass, field
from beartype import beartype
from typing import Generator, Union
import numpy as np
from faster_whisper.transcribe import Segment

from .sentence_ import Sentence
from .token_ import Token
from .param_ import Param
from .result_ import Result


@beartype
@dataclass
class Context:
    order: int = 0
    sc_offset: int = 0
    statistics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    prompt: Union[str, None] = None
    language: Union[str, None] = None
    completed: list[Sentence] = field(default_factory=list)
    candidate: list[Sentence] = field(default_factory=list)

    audio: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    processed_audio: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    timestamps: list[dict] = field(default_factory=list)

    prev_audio: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    prev_processed_audio: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    prev_timestamps: list[dict] = field(default_factory=list)
    prev_timestamps_mapping: list[dict[str, int]] = field(default_factory=list)
    prev_candidate_tokens: list[Token] = field(default_factory=list)
    prev_completed_tokens: list[Token] = field(default_factory=list)
    prev_candidate_sentences: list[Sentence] = field(default_factory=list)
    prev_sentence: Union[Sentence, None] = None

    merged_processed_audio: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    merged_timestamps: list[dict[str, int]] = field(default_factory=list)
    merged_timestamps_mapping: list[dict[str, int]] = field(default_factory=list)
    merged_candidate_tokens: Union[list[Token], Generator[Segment, None, None]] = field(
        default_factory=list
    )
    merged_completed_tokens: list[Token] = field(default_factory=list)
    merged_sentence: Union[Sentence, None] = None

    def get_default_value(self, field_name: str):
        field = self.__dataclass_fields__.get(field_name)
        if field is None:
            raise ValueError(f"No field named '{field_name}' in the dataclass.")

        if field.default_factory is not MISSING:
            return field.default_factory()
        elif field.default is not MISSING:
            return field.default
        else:
            raise ValueError(f"No default value for field '{field_name}'.")

    @property
    def audio_sc(self) -> int:
        return self.audio.shape[0]

    @property
    def processed_audio_sc(self) -> int:
        return self.processed_audio.shape[0]

    @property
    def prev_audio_sc(self) -> int:
        return self.prev_audio.shape[0]

    @property
    def prev_processed_audio_sc(self) -> int:
        return self.prev_processed_audio.shape[0]

    @property
    def merged_processed_audio_sc(self) -> int:
        return self.merged_processed_audio.shape[0]

    def bind(self, param: Param):
        self.completed = self.get_default_value("completed")
        self.candidate = self.get_default_value("candidate")
        self.merged_processed_audio = self.get_default_value("merged_processed_audio")
        self.merged_timestamps = self.get_default_value("merged_timestamps")
        self.merged_timestamps_mapping = self.get_default_value(
            "merged_timestamps_mapping"
        )
        self.merged_candidate_tokens = self.get_default_value("merged_candidate_tokens")
        self.merged_completed_tokens = self.get_default_value("merged_completed_tokens")
        self.merged_sentence = self.get_default_value("merged_sentence")

        self.order = param.order
        self.sc_offset = param.sc_offset
        self.statistics = param.statistics
        self.prompt = param.prompt
        self.language = param.language

        self.audio = self.processed_audio = param.audio
        self.timestamps = (
            [{"start": 0, "end": len(self.audio)}] if self.audio_sc > 0 else []
        )
        self.prev_audio = param.prev_audio
        self.prev_processed_audio = param.prev_processed_audio
        self.prev_timestamps = param.prev_timestamps
        self.prev_timestamps_mapping = param.prev_timestamps_mapping
        self.prev_completed_tokens = param.prev_completed_tokens
        self.prev_candidate_tokens = param.prev_candidate_tokens
        self.prev_candidate_sentences = param.prev_candidate_sentences
        self.prev_sentence = param.prev_sentence

    def extract(self):
        return Result(
            order=self.order,
            sc_offset=self.sc_offset,
            statistics=self.statistics,
            completed=self.completed,
            candidate=self.candidate,
            audio=self.audio,
            processed_audio=self.processed_audio,
            prev_audio=self.prev_audio,
            prev_processed_audio=self.prev_processed_audio,
            prev_timestamps=self.prev_timestamps,
            prev_timestamps_mapping=self.prev_timestamps_mapping,
            prev_candidate_tokens=self.prev_candidate_tokens,
            prev_completed_tokens=self.prev_completed_tokens,
            prev_sentence=self.prev_sentence,
        )
