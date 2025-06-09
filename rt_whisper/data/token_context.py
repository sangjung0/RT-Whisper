from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import MISSING, dataclass, field
import numpy as np

from .param import Param
from .result import Result
from .sentence import Sentence
from .token import Token

if TYPE_CHECKING:
    from rt_whisper.composer import ComposerStorage
    from rt_whisper.filters.duration_filter import DurationFilterStorage
    from rt_whisper.filters.probability_filter import ProbabilityFilterStorage
    from rt_whisper.processors.vad import VADStorage


# 임시 조치 #
def generate_composer_storage() -> ComposerStorage:
    from rt_whisper.composer import ComposerStorage

    return ComposerStorage()


def generate_vad_storage() -> VADStorage:
    from rt_whisper.processors.vad import VADStorage

    return VADStorage()


def generate_duration_filter_storage() -> DurationFilterStorage:
    from rt_whisper.filters.duration_filter import DurationFilterStorage

    return DurationFilterStorage()


def generate_probability_filter_storage() -> ProbabilityFilterStorage:
    from rt_whisper.filters.probability_filter import ProbabilityFilterStorage

    return ProbabilityFilterStorage()


# ## #

generate_empty_chunk = lambda: np.zeros((0,), dtype=np.float32)


@dataclass(slots=True)
class TokenContext:
    chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    prev_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    merged_chunk: np.ndarray = field(default_factory=generate_empty_chunk)

    order: int = field(default=0)
    offset: int = field(default=0)
    prompt: str | None = field(default=None)
    language: str | None = field(default=None)

    anchor_timestamp: int = field(default=0)

    segment_tokens: list[Token] = field(default_factory=list)
    prev_segment_tokens: list[Token] = field(default_factory=list)

    completed: list[Sentence] = field(default_factory=list)
    candidate: list[Sentence] = field(default_factory=list)

    vad: VADStorage = field(default_factory=generate_vad_storage)
    duration_filter: DurationFilterStorage = field(
        default_factory=generate_duration_filter_storage
    )
    probability_filter: ProbabilityFilterStorage = field(
        default_factory=generate_probability_filter_storage
    )
    composer: ComposerStorage = field(default_factory=generate_composer_storage)

    recycle_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    recycle_offset: int = field(default=0)
    recycle_segment_tokens: list[Token] = field(default_factory=list)

    # @property
    # def anchor_timestamp(self) -> int:
    #     return max(
    #         0, (self.prev_chunk.shape[0] + self.chunk.shape[0]) - self.OVERLAP_DURATION
    #     )

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

    def bind(self, param: Param):
        self.prev_chunk = param.recycle_chunk
        self.prev_segment_tokens = param.recycle_segment_tokens
        self.offset = param.offset

        self.chunk = param.chunk

        self.order = param.order
        self.prompt = param.prompt
        self.language = param.language

        self.vad.update(param.recycles.get("vad", None))
        self.duration_filter.update(param.recycles.get("duration_filter", None))
        self.probability_filter.update(param.recycles.get("probability_filter", None))
        self.composer.update(param.recycles.get("composer", None))

        self.merged_chunk = self.get_default_value("merged_chunk")
        self.segment_tokens = self.get_default_value("segment_tokens")
        self.completed = self.get_default_value("completed")
        self.candidate = self.get_default_value("candidate")

    def extract(self) -> Result:
        return Result(
            completed=self.completed,
            candidate=self.candidate,
            order=self.order,
            recycle_chunk=self.recycle_chunk,
            offset=self.recycle_offset,
            recycle_segment_tokens=self.recycle_segment_tokens,
            recycles={
                "vad": self.vad.extract(),
                "duration_filter": self.duration_filter.extract(),
                "probability_filter": self.probability_filter.extract(),
                "composer": self.composer.extract(),
            },
        )
