from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from rt_whisper.abstracts import Worker
from rt_whisper.data import Token

from .data import ASRParam, ASRResult

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext
    from faster_whisper.transcribe import Segment
    from typing import Any, Callable, Iterable


class ASRProcessor(Worker):
    def __init__(
        self,
        transcriber: Callable[[np.ndarray, str, str], tuple[Iterable, Any]],
        tokenizer_encode: Callable[[str], list[int]],
        sample_rate: int,
        within_eos: bool,
    ):
        super().__init__()
        self.__transcriber = transcriber
        self.__tokenizer_encode = tokenizer_encode
        self.__SAMPLE_RATE = sample_rate
        self.__WITHIN_EOS = within_eos

    # override
    def _can_process(self, context: TokenContext) -> ASRParam:
        return ASRParam.from_context(context)

    # override
    def _process(self, param: ASRParam) -> ASRResult:
        # 청크 합치기
        merged_chunk = np.concatenate([param.prev_chunk, param.chunk], axis=0)

        # 추론
        segments, language = self.__transcribe(
            merged_chunk, param.language, param.prompt
        )

        offset = param.offset - param.prev_chunk.shape[0]
        segment_tokens = self.__get_segment_tokens(
            segments,
            language,
            offset,
        )

        return ASRResult(
            merged_chunk=merged_chunk,
            segment_tokens=segment_tokens,
            language=language,
        )

    # override
    def _update(self, context: TokenContext, result: ASRResult) -> None:
        result.update_context(context)

    def __transcribe(self, chunk: np.ndarray, language: str | None, prompt: str | None):
        if chunk.shape[0] == 0:
            return [], language
        segments, info = self.__transcriber(chunk, language, prompt)
        language = info.language or language
        return segments, language

    def __get_segment_tokens(
        self,
        segments: list[Segment],
        language: str,
        offset: int,
    ):
        segment_tokens = []
        for segment in segments:
            tokens = [
                Token(
                    start=int(w.start * self.__SAMPLE_RATE) + offset,
                    end=int(w.end * self.__SAMPLE_RATE) + offset,
                    text=w.word,
                    lang=language,
                    tokens=self.__tokenizer_encode(w.word.lower()),
                    probability=w.probability,
                )
                for w in segment.words
            ]
            if self.__WITHIN_EOS:
                start = tokens[0].start
                end = tokens[-1].end
                tokens.append(
                    Token(
                        start=start,
                        end=end,
                        text=segment.text,
                        lang="",
                        tokens=[],
                        probability=1,
                        is_word=False,
                    )
                )
            segment_tokens.extend(tokens)

        return segment_tokens
