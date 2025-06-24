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


class VAD(Worker):
    def __init__(self, vad: Callable[[np.ndarray], list[dict[str, int]]]):
        super().__init__()
        self.__vad = vad

    # override
    def _register_state(self, state: TokenState):
        state.set_state(VADState, VADState())

    # override
    def _can_process(self, state: TokenState) -> VADProcessParam:
        asr_state: ASRState = state.get_state(ASRState)
        if state.chunk.shape[0] > 0 or asr_state.prev.chunk.shape[0] > 0:
            vad_state: VADState = state.get_state(VADState)
            vad_state.backup(state, asr_state)
            return VADProcessParam.from_state(vad_state)
        return None

    # override
    def _process(self, param: VADProcessParam) -> VADProcessResult:
        merged_chunk = np.concatenate([param.prev_chunk, param.chunk], axis=0)
        timestamps = vad(merged_chunk, self.__vad)
        merged_vad_chunk = merge_audio_from_timestamps(merged_chunk, timestamps)

        timestamps_mapping = generate_vad_timestamps_mapping(0, timestamps)

        prev_vad_chunk, vad_chunk = slice_vad_chunk(
            merged_vad_chunk, param.prev_chunk.shape[0], timestamps_mapping
        )

        return VADProcessResult(
            vad_chunk=vad_chunk,
            prev_vad_chunk=prev_vad_chunk,
            merged_chunk=merged_chunk,
            vad_offset=prev_vad_chunk.shape[0],
            vad_timestamps_mapping=timestamps_mapping,
        )

    # override
    def _update(self, state: TokenState, result: VADProcessResult) -> None:
        vad_state: VADState = state.get_state(VADState)
        asr_state: ASRState = state.get_state(ASRState)
        result.update_state(vad_state)
        vad_state.replace(state, asr_state)

    # override
    def _can_post_process(self, state: TokenState):
        vad_state: VADState = state.get_state(VADState)
        if (
            vad_state.original_chunk.shape[0] > 0
            or vad_state.original_prev_chunk.shape[0] > 0
        ):
            return VADPostParam.from_state(state, vad_state)
        return False

    # override
    def _post_process(self, param: VADPostParam):
        set_offset(param.segment_tokens, param.vad_timestamps_mapping)
        add_offset(param.segment_tokens, param.offset - param.prev_chunk.shape[0])

        return VADPostResult()

    # override
    def _post_update(self, state: TokenState, result: VADPostResult) -> None:
        vad_state: VADState = state.get_state(VADState)
        asr_state: ASRState = state.get_state(ASRState)
        result.update_state(vad_state)
        vad_state.post_replace(state, asr_state)
