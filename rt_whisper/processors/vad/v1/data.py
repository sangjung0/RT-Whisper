from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

from sj_utils.audio_utils import generate_empty_chunk

if TYPE_CHECKING:
    from rt_whisper.data import TokenState
    from rt_whisper.data import Token
    from rt_whisper.processors.asr import ASRState


@dataclass(slots=True)
class VADContext:
    chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    offset: int = field(default=0)
    timestamps: list[dict[str, int]] = field(default_factory=list)
    timestamps_mapping: list[dict[str, int]] = field(default_factory=list)


@dataclass(slots=True)
class VADState:
    # backup state
    chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    offset: int = field(default=0)
    prev_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    merged_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    original_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    original_offset: int = field(default=0)
    original_prev_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    original_merged_chunk: np.ndarray = field(default_factory=generate_empty_chunk)

    timestamps: list[dict[str, int]] = field(default_factory=list)
    timestamps_mapping: list[dict[str, int]] = field(default_factory=list)
    merged_timestamps: list[dict[str, int]] = field(default_factory=list)
    merged_timestamps_mapping: list[dict[str, int]] = field(default_factory=list)

    prev: VADContext = field(default_factory=VADContext)
    context: VADContext = field(default_factory=VADContext)

    def update(self, context: VADContext):
        assert isinstance(
            context, VADContext
        ), "recycle must be an instance of VADCache"

        self.__init__()
        self.prev = context

    def extract(self) -> VADContext:
        return self.context

    def backup(self, state: TokenState, asr_state: ASRState):
        self.original_chunk = state.chunk
        self.original_offset = state.offset
        self.original_prev_chunk = asr_state.prev.chunk

    def replace(self, state: TokenState, asr_state: ASRState):
        asr_state.prev.chunk = self.prev.chunk
        state.chunk = self.chunk
        state.offset = self.offset

    def post_backup(self, state: ASRState):
        self.merged_chunk = state.merged_chunk

    def post_replace(self, state: TokenState, asr_state: ASRState):
        state.chunk = self.original_chunk
        state.offset = self.original_offset

        asr_state.prev.chunk = self.original_prev_chunk
        asr_state.merged_chunk = self.original_merged_chunk


@dataclass(slots=True)
class VADProcessParam:
    chunk: np.ndarray
    offset: int
    prev_vad_offset: int
    prev_vad_chunk: np.ndarray

    @staticmethod
    def from_state(state: VADState):
        return VADProcessParam(
            chunk=state.original_chunk,
            offset=state.original_offset,
            prev_vad_offset=state.prev.offset,
            prev_vad_chunk=state.prev.chunk,
        )


@dataclass(slots=True)
class VADProcessResult:
    vad_chunk: np.ndarray
    vad_offset: int
    prev_vad_chunk: np.ndarray
    vad_timestamps: list[dict[str, int]]

    def update_state(self, state: VADState) -> None:
        state.chunk = self.vad_chunk
        state.offset = self.vad_offset
        state.timestamps = self.vad_timestamps


@dataclass(slots=True)
class VADPostParam:
    chunk: np.ndarray
    offset: int
    prev_chunk: np.ndarray
    vad_offset: int

    segment_tokens: list[Token]
    vad_timestamps: list[dict[str, int]]
    prev_vad_timestamps_mapping: list[dict[str, int]]

    @staticmethod
    def from_state(state: TokenState, vad_state: VADState):
        return VADPostParam(
            chunk=vad_state.original_chunk,
            offset=vad_state.original_offset,
            prev_chunk=vad_state.original_prev_chunk,
            vad_offset=vad_state.offset,
            segment_tokens=state.segment_tokens,
            vad_timestamps=vad_state.timestamps,
            prev_vad_timestamps_mapping=vad_state.prev.timestamps_mapping,
        )


@dataclass(slots=True)
class VADPostResult:
    original_merged_chunk: np.ndarray
    vad_timestamps_mapping: list[dict[str, int]]
    merged_vad_timestamps_mapping: list[dict[str, int]]
    # tokens: list[Token]

    def update_state(self, state: VADState):
        state.original_merged_chunk = self.original_merged_chunk
        state.timestamps_mapping = self.vad_timestamps_mapping
        state.merged_timestamps_mapping = self.merged_vad_timestamps_mapping
        # context.merged_candidate_tokens = tokens


@dataclass(slots=True)
class VADContextBuilderParam:
    anchor_timestamp: int
    vad_offset: int
    vad_chunk: np.ndarray
    vad_timestamps: list[dict[str, int]]
    prev_vad_timestamps: list[dict[str, int]]
    merged_vad_chunk: np.ndarray
    merged_vad_timestamps_mapping: list[dict[str, int]]

    @staticmethod
    def from_state(state: TokenState, vad_state: VADState):
        return VADContextBuilderParam(
            anchor_timestamp=state.anchor_timestamp,
            vad_offset=vad_state.offset,
            vad_chunk=vad_state.chunk,
            vad_timestamps=vad_state.timestamps,
            prev_vad_timestamps=vad_state.prev.timestamps,
            merged_vad_chunk=vad_state.merged_chunk,
            merged_vad_timestamps_mapping=vad_state.merged_timestamps_mapping,
        )


@dataclass(slots=True)
class VADContextBuilderResult:
    context_vad_offset: int
    context_vad_chunk: np.ndarray
    context_vad_timestamps: list[dict[str, int]]
    context_vad_timestamps_mapping: list[dict[str, int]]

    def update_context(self, state: VADState):
        state.context.chunk = self.context_vad_chunk
        state.context.offset = self.context_vad_offset
        state.context.timestamps = self.context_vad_timestamps
        state.context.timestamps_mapping = self.context_vad_timestamps_mapping
