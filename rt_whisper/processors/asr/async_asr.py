from __future__ import annotations
from typing import Any, Callable, Iterable, TYPE_CHECKING
import numpy as np

from rt_whisper.abstracts import AsyncWorker

from .data import *
from .mixin import *
from .service import *

if TYPE_CHECKING:
    from rt_whisper.data import TokenState
    from typing import Any, Callable, Iterable


class AsyncASRProcessor(ASRProcessorMixin, AsyncWorker):
    def __init__(
        self,
        transcriber: Callable[[np.ndarray, str, str], tuple[Iterable, Any]],
        tokenizer_encoder: Callable[[str], list[int]],
        sample_rate: int,
        within_eos: bool,
    ):
        super().__init__()
        self._transcriber = transcriber
        self._tokenizer_encoder = tokenizer_encoder
        self._SAMPLE_RATE = sample_rate
        self._WITHIN_EOS = within_eos

    async def _process(self, param: ASRParam) -> ASRResult:
        # 청크 합치기
        merged_chunk = np.concatenate([param.prev_chunk, param.chunk], axis=0)

        # 추론
        segments, language = await async_transcribe(
            merged_chunk, param.language, self._transcriber
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


class AsyncASRContextBuilder(ASRContextBuilderMixin, AsyncASRProcessor):
    def __init__(
        self,
        *args,
        max_overlap_duration: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._MAX_OVERLAP_DURATION = max_overlap_duration

    async def _context_build(
        self, param: ASRContextBuilderParam
    ) -> ASRContextBuilderResult:
        return super()._context_build(param)


class AsyncASR(AsyncASRContextBuilder):
    def __init__(
        self,
        *args,
        transcriber: Callable[[np.ndarray, str, str], tuple[Iterable, Any]],
        tokenizer_encoder: Callable[[str], list[int]],
        sample_rate: int,
        within_eos: bool,
        **kwargs,
    ):
        super().__init__(
            *args,
            transcriber=transcriber,
            tokenizer_encoder=tokenizer_encoder,
            sample_rate=sample_rate,
            within_eos=within_eos,
            **kwargs,
        )

    # override
    def _register_state(self, state: TokenState) -> None:
        state.set_state(ASRState, ASRState())
