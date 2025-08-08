from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from dataclasses import dataclass, field
from typing import Union

from sj_utils.audio import generate_empty_chunk

if TYPE_CHECKING:
    from rt_whisper.data import Token, TokenState


@dataclass(slots=True)
class ASRContext:
    chunk: np.ndarray = field(default_factory=generate_empty_chunk)


@dataclass(slots=True)
class ASRState:
    merged_chunk: np.ndarray = field(default_factory=generate_empty_chunk)

    prev: ASRContext = field(default_factory=ASRContext)
    context: ASRContext = field(default_factory=ASRContext)

    def update(self, context: ASRContext) -> None:
        assert isinstance(
            context, ASRContext
        ), "context must be an instance of ASRContext"

        self.__init__()
        self.prev = context

    def extract(self) -> ASRContext:
        return self.context


@dataclass(slots=True)
class ASRParam:
    chunk: np.ndarray
    offset: int
    language: Union[str, None]
    prompt: Union[str, None]
    prev_chunk: np.ndarray

    @staticmethod
    def from_state(state: TokenState, asr_state: ASRState) -> "ASRParam":
        return ASRParam(
            chunk=state.chunk,
            offset=state.offset,
            language=state.language,
            prompt=state.prompt,
            prev_chunk=asr_state.prev.chunk,
        )


@dataclass(slots=True)
class ASRResult:
    merged_chunk: np.ndarray
    segment_tokens: list[Token]
    language: str | None

    def update_state(self, state: TokenState, asr_state: ASRState) -> None:
        asr_state.merged_chunk = self.merged_chunk
        state.segment_tokens = self.segment_tokens
        state.language = self.language


@dataclass(slots=True)
class ASRContextBuilderParam:
    chunk: np.ndarray
    offset: int
    merged_chunk: np.ndarray
    prev_chunk: np.ndarray
    segment_tokens: list[Token] = field(default_factory=list)

    @staticmethod
    def from_state(state: TokenState, asr_state: ASRState) -> "ASRContextBuilderParam":
        return ASRContextBuilderParam(
            chunk=state.chunk,
            offset=state.offset,
            merged_chunk=asr_state.merged_chunk,
            prev_chunk=asr_state.prev.chunk,
            segment_tokens=state.segment_tokens,
        )


@dataclass(slots=True)
class ASRContextBuilderResult:
    context_chunk: np.ndarray
    context_offset: int
    anchor_timestamp: int

    def update_state(self, state: TokenState, asr_state: ASRState) -> None:
        asr_state.context.chunk = self.context_chunk
        state.offset = self.context_offset
        state.anchor_timestamp = self.anchor_timestamp


__all__ = [
    "ASRContext",
    "ASRState",
    "ASRParam",
    "ASRResult",
    "ASRContextBuilderParam",
    "ASRContextBuilderResult",
]
