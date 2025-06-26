from __future__ import annotations
from typing import Any, Callable, Iterable, TYPE_CHECKING, Protocol
import numpy as np

from .data import *
from .service import *

if TYPE_CHECKING:
    from rt_whisper.data import TokenState
    from typing import Any, Callable, Iterable


class ASRProcessorDeps(Protocol):
    _transcriber: Callable[[np.ndarray, str, str], tuple[Iterable, Any]]
    _tokenizer_encoder: Callable[[str], list[int]]
    _SAMPLE_RATE: int
    _WITHIN_EOS: bool


class ASRProcessorMixin:
    def _can_process(self: ASRProcessorDeps, state: TokenState) -> ASRParam:
        asr_state: ASRState = state.get_state(ASRState)
        return ASRParam.from_state(state, asr_state)

    def _process(self: ASRProcessorDeps, param: ASRParam) -> ASRResult:
        # 청크 합치기
        merged_chunk = np.concatenate([param.prev_chunk, param.chunk], axis=0)

        # 추론
        segments, language = transcribe(
            merged_chunk, param.language, param.prompt, self._transcriber
        )

        segment_tokens = segment_to_token_list(
            segments,
            language,
            param.offset - param.prev_chunk.shape[0],
            self._SAMPLE_RATE,
            self._WITHIN_EOS,
            self._tokenizer_encoder,
        )

        return ASRResult(
            merged_chunk=merged_chunk,
            segment_tokens=segment_tokens,
            language=language,
        )

    def _update(self: ASRProcessorDeps, state: TokenState, result: ASRResult) -> None:
        asr_state: ASRState = state.get_state(ASRState)
        result.update_state(state, asr_state)


class ASRContextBuilderDeps(Protocol):
    _MAX_OVERLAP_DURATION: int


class ASRContextBuilderMixin:

    def _can_build(
        self: ASRContextBuilderDeps, context: TokenState
    ) -> ASRContextBuilderParam:
        asr_state: ASRState = context.get_state(ASRState)
        return ASRContextBuilderParam.from_state(context, asr_state)

    def _context_build(
        self: ASRContextBuilderDeps, param: ASRContextBuilderParam
    ) -> ASRContextBuilderResult:
        context_chunk, context_offset, anchor_timestamp = generate_overlap_context(
            merged_chunk=param.merged_chunk,
            current_chunk=param.chunk,
            prev_chunk=param.prev_chunk,
            offset=param.offset,
            max_overlap_duration=self.__MAX_OVERLAP_DURATION,
        )

        anchor_timestamp = adjust_anchor_timestamp(
            anchor_timestamp, param.segment_tokens
        )

        return ASRContextBuilderResult(
            context_chunk=context_chunk,
            context_offset=context_offset,
            anchor_timestamp=anchor_timestamp,
        )

    def _context_update(
        self: ASRContextBuilderDeps, state: TokenState, result: ASRContextBuilderResult
    ) -> None:
        asr_state: ASRState = state.get_state(ASRState)
        result.update_state(state, asr_state)
