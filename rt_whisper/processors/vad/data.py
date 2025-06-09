from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext
    from rt_whisper.data import Token

generate_empty_chunk = lambda: np.zeros((0,), dtype=np.float32)


@dataclass(slots=True)
class VADProcessParam:
    chunk: np.ndarray
    offset: int
    prev_vad_offset: int
    prev_vad_chunk: np.ndarray

    @staticmethod
    def from_context(context: TokenContext):
        context.vad.original_chunk = context.chunk
        context.vad.original_offset = context.offset
        context.vad.original_prev_chunk = context.prev_chunk
        return VADProcessParam(
            chunk=context.chunk,
            offset=context.offset,
            prev_vad_offset=context.vad.prev.vad_offset,
            prev_vad_chunk=context.vad.prev.vad_chunk,
        )


@dataclass(slots=True)
class VADProcessResult:
    vad_chunk: np.ndarray
    vad_offset: int
    prev_vad_chunk: np.ndarray
    vad_timestamps: list[dict[str, int]]

    def update_context(self, context: TokenContext) -> None:
        context.vad.chunk = self.vad_chunk
        context.vad.offset = self.vad_offset
        context.vad.timestamps = self.vad_timestamps

        context.chunk = self.vad_chunk  # 기존 chunk를 대체
        context.offset = self.vad_offset
        context.prev_chunk = self.prev_vad_chunk


@dataclass(slots=True)
class VADPostParam:
    chunk: np.ndarray
    prev_chunk: np.ndarray

    segment_tokens: list[Token]
    vad_timestamps: list[dict[str, int]]
    prev_vad_timestamps_mapping: list[dict[str, int]]

    @staticmethod
    def from_context(context: TokenContext):
        context.vad.merged_chunk = context.merged_chunk
        return VADPostParam(
            chunk=context.vad.original_chunk,
            prev_chunk=context.vad.original_prev_chunk,
            segment_tokens=context.segment_tokens,
            vad_timestamps=context.vad.timestamps,
            prev_vad_timestamps_mapping=context.vad.prev.timestamps_mapping,
        )


@dataclass(slots=True)
class VADPostResult:
    merged_chunk: np.ndarray
    vad_timestamps_mapping: list[dict[str, int]]
    merged_vad_timestamps_mapping: list[dict[str, int]]
    # tokens: list[Token]

    def update_context(self, context: TokenContext):
        context.chunk = context.vad.original_chunk
        context.prev_chunk = context.vad.original_prev_chunk
        context.offset = context.vad.original_offset

        context.merged_chunk = self.merged_chunk
        context.vad.timestamps_mapping = self.vad_timestamps_mapping
        context.vad.merged_timestamps_mapping = self.merged_vad_timestamps_mapping
        # context.merged_candidate_tokens = tokens


@dataclass(slots=True)
class VADRecycleParam:
    anchor_timestamp: int
    vad_offset: int
    vad_chunk: np.ndarray
    vad_timestamps: list[dict[str, int]]
    prev_vad_timestamps: list[dict[str, int]]
    merged_vad_timestamps_mapping: list[dict[str, int]]

    @staticmethod
    def from_context(context: TokenContext):
        return VADRecycleParam(
            anchor_timestamp=context.anchor_timestamp,
            vad_offset=context.vad.offset,
            vad_chunk=context.vad.chunk,
            vad_timestamps=context.vad.timestamps,
            prev_vad_timestamps=context.vad.prev.timestamps,
            merged_vad_timestamps_mapping=context.vad.merged_timestamps_mapping,
        )


@dataclass(slots=True)
class VADRecycleResult:
    recycle_vad_offset: int
    recycle_vad_chunk: np.ndarray
    recycle_vad_timestamps: list[dict[str, int]]
    recycle_vad_timestamps_mapping: list[dict[str, int]]

    def update_context(self, context: TokenContext):
        context.vad.recycle.vad_chunk = self.recycle_vad_chunk
        context.vad.recycle.vad_offset = self.recycle_vad_offset
        context.vad.recycle.timestamps = self.recycle_vad_timestamps
        context.vad.recycle.timestamps_mapping = self.recycle_vad_timestamps_mapping


@dataclass(slots=True)
class VADRecycle:
    vad_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    vad_offset: int = field(default=0)
    timestamps: list[dict[str, int]] = field(default_factory=list)
    timestamps_mapping: list[dict[str, int]] = field(default_factory=list)


@dataclass(slots=True)
class VADStorage:
    chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    offset: int = field(default=0)
    merged_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    original_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    original_offset: int = field(default=0)
    original_prev_chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    timestamps: list[dict[str, int]] = field(default_factory=list)
    timestamps_mapping: list[dict[str, int]] = field(default_factory=list)
    merged_timestamps: list[dict[str, int]] = field(default_factory=list)
    merged_timestamps_mapping: list[dict[str, int]] = field(default_factory=list)

    prev: VADRecycle = field(default_factory=VADRecycle)
    recycle: VADRecycle = field(default_factory=VADRecycle)

    def update(self, recycle: VADRecycle):
        self.__init__()
        if recycle is not None:
            self.prev = recycle

    def extract(self) -> VADRecycle:
        return self.recycle
