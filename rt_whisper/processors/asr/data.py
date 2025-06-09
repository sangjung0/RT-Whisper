from __future__ import annotations
from typing import Iterator, TYPE_CHECKING
from dataclasses import dataclass
from typing import Iterator, Union
import numpy as np


if TYPE_CHECKING:
    from rt_whisper.data import Token, TokenContext


@dataclass(slots=True)
class ASRParam:
    chunk: np.ndarray
    prev_chunk: np.ndarray
    offset: int
    language: Union[str, None]
    prompt: Union[str, None]

    @staticmethod
    def from_context(context: TokenContext) -> "ASRParam":
        return ASRParam(
            chunk=context.chunk,
            prev_chunk=context.prev_chunk,
            offset=context.offset,
            language=context.language,
            prompt=context.prompt,
        )


@dataclass(slots=True)
class ASRResult:
    merged_chunk: np.ndarray
    segment_tokens: Iterator[Token]
    language: str | None

    def update_context(self, context: TokenContext) -> None:
        context.merged_chunk = self.merged_chunk
        context.segment_tokens = self.segment_tokens
        context.language = self.language


@dataclass(slots=True)
class ASRRecycleParam:
    chunk: np.ndarray
    merged_chunk: np.ndarray
    prev_chunk: np.ndarray
    offset: int

    @staticmethod
    def from_context(context: TokenContext) -> "ASRRecycleParam":
        return ASRRecycleParam(
            chunk=context.chunk,
            merged_chunk=context.merged_chunk,
            prev_chunk=context.prev_chunk,
            offset=context.offset,
        )


@dataclass(slots=True)
class ASRRecycleResult:
    recycle_chunk: np.ndarray
    recycle_offset: int
    anchor_timestamp: int

    def update_context(self, context: TokenContext) -> None:
        context.recycle_chunk = self.recycle_chunk
        context.recycle_offset = self.recycle_offset
        context.anchor_timestamp = self.anchor_timestamp
