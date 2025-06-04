from dataclasses import MISSING, dataclass, field
from beartype import beartype
from typing import Union
import numpy as np

from .sentence import Sentence
from .token import Token
from .param import Param
from .result import Result

empty_chunk = lambda: np.zeros((0,), dtype=np.float32)


@beartype
@dataclass
class Context:
    # TODO worker간의 의존성 분리가 되면, 이 부분도 수정 필요함

    order: int = field(default=0)  # using: composer | output: composer
    offset: int = field(default=0)  # using: vad token_classifier
    statistics: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )  # using: duration_filter_param | output: duration_filter_param | using: probability_filter_param | output: probability_filter_param
    prompt: Union[str, None] = field(default=None)  # using asr
    language: Union[str, None] = field(
        default=None
    )  # using: asr | output: asr | using: duration_filter_param probability_filter_param selector composer

    chunk: np.ndarray = field(
        default_factory=empty_chunk
    )  # using: vad token_classifier
    vad_chunk: np.ndarray = field(
        default_factory=empty_chunk
    )  # output: vad | using: asr
    vad_timestamps: list[dict] = field(default_factory=list)  # output: vad | using: vad
    vad_timestamps_mapping: list[dict[str, int]] = field(
        default_factory=list
    )  # output: vad

    prev_chunk: np.ndarray = field(
        default_factory=empty_chunk
    )  # using: token_classifier
    prev_vad_chunk: np.ndarray = field(default_factory=empty_chunk)  # using: asr
    prev_vad_timestamps: list[dict] = field(
        default_factory=list
    )  # using: token_classifier
    prev_vad_timestamps_mapping: list[dict[str, int]] = field(
        default_factory=list
    )  # using: vad
    prev_candidate_tokens: list[Token] = field(default_factory=list)  # using: selector
    prev_completed_tokens: list[Token] = field(
        default_factory=list
    )  # using: token_classifier

    merged_vad_chunk: np.ndarray = field(
        default_factory=empty_chunk
    )  # output: asr | using: token_classifier
    merged_vad_timestamps_mapping: list[dict[str, int]] = field(
        default_factory=list
    )  # output: vad | using: token_classifier

    candidate_tokens: list[Token] = field(
        default_factory=list
    )  # output: asr | using: vad position_weighted_filter duration_filter_param | output: position_weighted_filter | using: probability_filter_param | output: probability_filter_params | using: selector | output: selector | using: token_classifier composer
    completed_tokens: list[Token] = field(
        default_factory=list
    )  # output: token_classifier | using: composer

    completed: list[Sentence] = field(default_factory=list)  # output: composer
    candidate: list[Sentence] = field(default_factory=list)  # output: composer

    next_offset: int = field(default=0)  # output: token_classifier
    recycle_chunk: np.ndarray = field(
        default_factory=empty_chunk
    )  # output: token_classifier
    recycle_vad_chunk: np.ndarray = field(
        default_factory=empty_chunk
    )  # output: token_classifier
    recycle_vad_timestamps: list[dict[str, int]] = field(
        default_factory=list
    )  # output: token_classifier
    recycle_vad_timestamps_mapping: list[dict[str, int]] = field(
        default_factory=list
    )  # output: token_classifier
    recycle_candidate_tokens: list[Token] = field(
        default_factory=list
    )  # output: token_classifier
    recycle_completed_tokens: list[Token] = field(
        default_factory=list
    )  # output: composer

    prev_candidate_sentences: list[Sentence] = field(default_factory=list)
    prev_sentence: Union[Sentence, None] = field(default=None)
    merged_sentence: Union[Sentence, None] = field(default=None)

    @property
    def chunk_size(self) -> int:  # using: vad position_weighted_filter token_classifier
        return self.chunk.shape[0]

    @property
    def vad_chunk_size(self) -> int:  # using: asr
        return self.vad_chunk.shape[0]

    @property
    def prev_chunk_size(
        self,
    ) -> int:  # using: position_weighted_filter token_classifier
        return self.prev_chunk.shape[0]

    @property
    def merged_vad_chunk_size(
        self,
    ) -> int:  # using: position_weighted_filter token_classifier
        return self.merged_vad_chunk.shape[0]

    @property
    def prev_chunk_offset(
        self,
    ) -> int:  # using: asr position_weighted_filter token_classifier
        return self.offset - self.prev_chunk_size

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
        self.order = param.order
        self.offset = param.offset
        self.statistics = param.statistics
        self.prompt = param.prompt
        self.language = param.language

        self.chunk = param.chunk
        self.vad_chunk = self.get_default_value("vad_chunk")
        self.vad_timestamps = (
            [{"start": 0, "end": len(self.chunk)}] if self.chunk.size > 0 else []
        )  # 일단 두자
        self.vad_timestamps_mapping = self.get_default_value("vad_timestamps_mapping")

        self.prev_chunk = param.recycle_chunk
        self.prev_vad_chunk = param.recycle_vad_chunk
        self.prev_vad_timestamps = param.recycle_vad_timestamps
        self.prev_vad_timestamps_mapping = param.recycle_vad_timestamps_mapping
        self.prev_completed_tokens = param.recycle_completed_tokens
        self.prev_candidate_tokens = param.recycle_candidate_tokens

        self.merged_vad_chunk = self.get_default_value("merged_vad_chunk")
        self.merged_vad_timestamps_mapping = self.get_default_value(
            "merged_vad_timestamps_mapping"
        )
        self.candidate_tokens = self.get_default_value("candidate_tokens")
        self.completed_tokens = self.get_default_value("completed_tokens")
        self.completed = self.get_default_value("completed")
        self.candidate = self.get_default_value("candidate")

        self.recycle_chunk = self.get_default_value("recycle_chunk")
        self.recycle_vad_chunk = self.get_default_value("recycle_vad_chunk")
        self.recycle_vad_timestamps = self.get_default_value("recycle_vad_timestamps")
        self.recycle_vad_timestamps_mapping = self.get_default_value(
            "recycle_vad_timestamps_mapping"
        )
        self.recycle_candidate_tokens = self.get_default_value(
            "recycle_candidate_tokens"
        )
        self.recycle_completed_tokens = self.get_default_value(
            "recycle_completed_tokens"
        )

        self.prev_candidate_sentences = param.prev_candidate_sentences
        self.prev_sentence = param.prev_sentence
        self.merged_sentence = self.get_default_value("merged_sentence")

    def extract(self):
        return Result(
            order=self.order,
            next_offset=self.offset,
            statistics=self.statistics,
            completed=self.completed,
            candidate=self.candidate,
            chunk=self.chunk,
            vad_chunk=self.vad_chunk,
            recycle_chunk=self.recycle_chunk,
            recycle_vad_chunk=self.recycle_vad_chunk,
            recycle_vad_timestamps=self.recycle_vad_timestamps,
            recycle_vad_timestamps_mapping=self.recycle_vad_timestamps_mapping,
            recycle_candidate_tokens=self.recycle_candidate_tokens,
            recycle_completed_tokens=self.recycle_completed_tokens,
            prev_sentence=self.prev_sentence,
        )
