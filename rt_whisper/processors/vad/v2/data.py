from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from dataclasses import dataclass, field

from sj_utils.audio_utils import generate_empty_chunk

if TYPE_CHECKING:
    from rt_whisper.data import TokenState
    from rt_whisper.data import Token
    from rt_whisper.processors.asr import ASRState


@dataclass(slots=True)
class VADState:
    # backup state
    chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    offset: int = field(default=0)
    prev_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    original_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    original_offset: int = field(default=0)
    original_prev_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    original_merged_chunk: np.ndarray = field(default_factory=generate_empty_chunk)

    timestamps_mapping: list[dict[str, int]] = field(default_factory=list)

    def update(self, *args, **kwargs):
        self.__init__()

    def extract(self) -> VADState:
        return None

    def backup(self, state: TokenState, asr_state: ASRState):
        self.original_chunk = state.chunk
        self.original_offset = state.offset
        self.original_prev_chunk = asr_state.prev.chunk

    def replace(self, state: TokenState, asr_state: ASRState):
        state.chunk = self.chunk
        state.offset = self.offset
        asr_state.prev.chunk = self.prev_chunk

    def post_replace(self, state: TokenState, asr_state: ASRState):
        state.chunk = self.original_chunk
        state.offset = self.original_offset

        asr_state.prev.chunk = self.original_prev_chunk
        asr_state.merged_chunk = self.original_merged_chunk


@dataclass(slots=True)
class VADProcessParam:
    chunk: np.ndarray
    prev_chunk: np.ndarray

    @staticmethod
    def from_state(state: VADState) -> VADProcessParam:
        return VADProcessParam(
            chunk=state.original_chunk,
            prev_chunk=state.original_prev_chunk,
        )


@dataclass(slots=True)
class VADProcessResult:
    vad_chunk: np.ndarray
    prev_vad_chunk: np.ndarray
    merged_chunk: np.ndarray
    vad_offset: int
    vad_timestamps_mapping: list[dict[str, int]]

    def update_state(self, state: VADState) -> None:
        state.chunk = self.vad_chunk
        state.prev_chunk = self.prev_vad_chunk
        state.original_merged_chunk = self.merged_chunk
        state.offset = self.vad_offset
        state.timestamps_mapping = self.vad_timestamps_mapping


@dataclass(slots=True)
class VADPostParam:
    offset: int
    prev_chunk: np.ndarray
    vad_timestamps_mapping: list[dict[str, int]]

    segment_tokens: list[Token]

    @staticmethod
    def from_state(state: TokenState, vad_state: VADState) -> VADPostParam:
        return VADPostParam(
            offset=vad_state.original_offset,
            prev_chunk=vad_state.original_prev_chunk,
            vad_timestamps_mapping=vad_state.timestamps_mapping,
            segment_tokens=state.segment_tokens,
        )


@dataclass(slots=True)
class VADPostResult:
    # merged_chunk: np.ndarray

    def update_state(self, state: VADState) -> None:
        ...
        # state.original_merged_chunk = self.merged_chunk
