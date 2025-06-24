from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from rt_whisper.abstracts import Worker
from rt_whisper.processors.asr import ASRState
from rt_whisper.processors.vad.common.service import *

from .data import *
from .service import *

if TYPE_CHECKING:
    from typing import Callable
    from rt_whisper.data import TokenState


class VADProcessor(Worker):
    def __init__(self, vad: Callable[[np.ndarray], list[dict[str, int]]]):
        super().__init__()
        self.__vad = vad

    # override
    def _can_process(self, state: TokenState) -> VADProcessParam:
        vad_state: VADState = state.get_state(VADState)
        asr_state: ASRState = state.get_state(ASRState)
        vad_state.backup(state, asr_state)
        return VADProcessParam.from_state(vad_state)

    # override
    def _process(self, param: VADProcessParam) -> VADProcessResult:
        vad_offset = generate_vad_offset(param.prev_vad_offset, param.prev_vad_chunk)
        timestamps = vad(param.chunk, self.__vad)
        vad_chunk = merge_audio_from_timestamps(param.chunk, timestamps)
        adjusted_timestamps = apply_offset_to_timestamps(timestamps, param.offset)

        return VADProcessResult(
            vad_chunk=vad_chunk,
            vad_offset=vad_offset,
            prev_vad_chunk=param.prev_vad_chunk,
            vad_timestamps=adjusted_timestamps,
        )

    # override
    def _update(self, state: TokenState, result: VADProcessResult) -> None:
        vad_state: VADState = state.get_state(VADState)
        asr_state: ASRState = state.get_state(ASRState)
        result.update_state(vad_state)
        vad_state.replace(state, asr_state)


class VADPostProcessor(VADProcessor):
    # override
    def _can_post_process(self, state: TokenState) -> VADPostParam:
        vad_state: VADState = state.get_state(VADState)
        asr_state: ASRState = state.get_state(ASRState)
        vad_state.post_backup(asr_state)
        return VADPostParam.from_state(state, vad_state)

    # override
    def _post_process(self, param: VADPostParam) -> VADPostResult:
        # 청크 합치기
        merged_chunk = np.concatenate([param.prev_chunk, param.chunk], axis=0)

        prev_end = (
            param.prev_vad_timestamps_mapping[-1]["end"]
            if param.prev_vad_timestamps_mapping
            else param.vad_offset
        )

        vad_timestamps_mapping = generate_vad_timestamps_mapping(
            prev_end,
            param.vad_timestamps,
        )

        merged_vad_timestamps_mapping = (
            param.prev_vad_timestamps_mapping + vad_timestamps_mapping
        )

        set_offset(
            param.segment_tokens,
            merged_vad_timestamps_mapping,
        )

        return VADPostResult(
            original_merged_chunk=merged_chunk,
            vad_timestamps_mapping=vad_timestamps_mapping,
            merged_vad_timestamps_mapping=merged_vad_timestamps_mapping,
        )

    # override
    def _post_update(self, state: TokenState, result: VADPostResult):
        vad_state: VADState = state.get_state(VADState)
        asr_state: ASRState = state.get_state(ASRState)
        result.update_state(vad_state)
        vad_state.post_replace(state, asr_state)


class VADContextBuilder(VADPostProcessor):
    # override
    def _can_build(self, state: TokenState) -> VADContextBuilderParam:
        vad_state: VADState = state.get_state(VADState)
        return VADContextBuilderParam.from_state(state, vad_state)

    # override
    def _context_build(self, param: VADContextBuilderParam):
        context_vad_timestamps = generate_context_vad_timestamps(
            param.prev_vad_timestamps,
            param.vad_timestamps,
            param.anchor_timestamp,
        )

        context_vad_timestamps_mapping = generate_context_vad_timestamps_mapping(
            param.merged_vad_timestamps_mapping,
            param.anchor_timestamp,
        )

        context_vad_chunk = generate_context_vad_chunk(
            context_vad_timestamps_mapping, param.merged_vad_chunk
        )

        context_vad_offset = generate_context_vad_offset(
            param.vad_offset,
            param.vad_chunk,
            context_vad_chunk,
        )

        return VADContextBuilderResult(
            context_vad_offset=context_vad_offset,
            context_vad_chunk=context_vad_chunk,
            context_vad_timestamps=context_vad_timestamps,
            context_vad_timestamps_mapping=context_vad_timestamps_mapping,
        )

    # override
    def _context_update(
        self, state: TokenState, result: VADContextBuilderResult
    ) -> None:
        vad_state: VADState = state.get_state(VADState)
        result.update_context(vad_state)


class VAD(VADContextBuilder):
    def __init__(
        self, vad: Callable[[np.ndarray], list[dict[str, int]]], *args, **kwargs
    ):
        super().__init__(vad, *args, **kwargs)

    # override
    def _register_state(self, state: TokenState):
        state.set_state(VADState, VADState())
