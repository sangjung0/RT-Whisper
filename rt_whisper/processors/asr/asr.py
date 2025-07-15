from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np

from logging import Logger
from typing import Any, Callable, Iterable

from rt_whisper.abstracts import Worker
from rt_whisper.processors.asr.data import (
    ASRParam,
    ASRResult,
    ASRState,
    ASRContextBuilderParam,
    ASRContextBuilderResult,
)
from rt_whisper.processors.asr.service import (
    transcribe,
    segment_to_token_list,
    generate_overlap_context,
    adjust_anchor_timestamp,
)


if TYPE_CHECKING:
    from rt_whisper.data import TokenState


class ASRProcessor(Worker):
    def __init__(
        self,
        transcriber: Callable[[np.ndarray, str, str], tuple[Iterable, Any]],
        embed: Callable[[str], torch.Tensor],
        logger: Logger,
        sample_rate: int,
        within_eos: bool,
    ):
        super().__init__()
        self.logger = logger

        self.__transcriber = transcriber
        self.__embed = embed
        self.__SAMPLE_RATE = sample_rate
        self.__WITHIN_EOS = within_eos

    # override
    def _can_process(self, state: TokenState) -> ASRParam:
        asr_state: ASRState = state.get_state(ASRState)
        return ASRParam.from_state(state, asr_state)

    # override
    def _process(self, param: ASRParam) -> ASRResult:
        self.logger.debug(f"\tProcessing ASR")

        # 청크 합치기
        merged_chunk = np.concatenate([param.prev_chunk, param.chunk], axis=0)
        self.logger.debug(
            f"\t\tMerged chunk: {param.prev_chunk.shape} + {param.chunk.shape} = {merged_chunk.shape}"
        )

        # 추론
        segments, language = transcribe(
            merged_chunk, param.language, param.prompt, self.__transcriber
        )

        segment_tokens = segment_to_token_list(
            segments,
            language,
            param.offset - param.prev_chunk.shape[0],
            self.__SAMPLE_RATE,
            self.__WITHIN_EOS,
            self.__embed,
        )
        self.logger.debug(f"\t\tSegment tokens: {segment_tokens}")

        return ASRResult(
            merged_chunk=merged_chunk,
            segment_tokens=segment_tokens,
            language=language,
        )

    # override
    def _update(self, state: TokenState, result: ASRResult) -> None:
        asr_state: ASRState = state.get_state(ASRState)
        result.update_state(state, asr_state)


class ASRContextBuilder(ASRProcessor):
    def __init__(
        self,
        *args,
        max_overlap_duration: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__MAX_OVERLAP_DURATION = max_overlap_duration

    # override
    def _can_build(self, context: TokenState):
        asr_state: ASRState = context.get_state(ASRState)
        return ASRContextBuilderParam.from_state(context, asr_state)

    # override
    def _context_build(self, param: ASRContextBuilderParam) -> ASRContextBuilderResult:
        self.logger.debug(f"\tBuilding ASR context")

        context_chunk, context_offset, anchor_timestamp = generate_overlap_context(
            merged_chunk=param.merged_chunk,
            current_chunk=param.chunk,
            prev_chunk=param.prev_chunk,
            offset=param.offset,
            max_overlap_duration=self.__MAX_OVERLAP_DURATION,
        )
        self.logger.debug(
            f"\t\tcontext_chunk: {context_chunk.shape}"
            f"\t\tcontext_offset: {context_offset}"
            f"\t\tanchor_timestamp: {anchor_timestamp}"
        )

        anchor_timestamp = adjust_anchor_timestamp(
            anchor_timestamp, param.segment_tokens
        )
        self.logger.debug(f"\t\tAdjusted anchor timestamp: {anchor_timestamp}")

        return ASRContextBuilderResult(
            context_chunk=context_chunk,
            context_offset=context_offset,
            anchor_timestamp=anchor_timestamp,
        )

    # override
    def _context_update(
        self, state: TokenState, result: ASRContextBuilderResult
    ) -> None:
        asr_state: ASRState = state.get_state(ASRState)
        result.update_state(state, asr_state)


class ASR(ASRContextBuilder):
    def __init__(
        self,
        *args,
        transcriber: Callable[[np.ndarray, str, str], tuple[Iterable, Any]],
        embed: Callable[[str], torch.Tensor],
        sample_rate: int,
        within_eos: bool,
        max_overlap_duration: int,
        logger: Logger,
        **kwargs,
    ):
        super().__init__(
            *args,
            transcriber=transcriber,
            embed=embed,
            sample_rate=sample_rate,
            within_eos=within_eos,
            max_overlap_duration=max_overlap_duration,
            logger=logger,
            **kwargs,
        )

    # override
    def _register_state(self, state: TokenState) -> None:
        state.set_state(ASRState, ASRState())
