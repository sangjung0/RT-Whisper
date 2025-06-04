from typing import Any, Callable, Iterable
import numpy as np

from rt_whisper.abstracts import Worker
from rt_whisper.data import Context, Token

from .data import ASRParam, ASRResult


class ASR(Worker):
    def __init__(
        self,
        transcribe: Callable[[np.ndarray, str, str], tuple[Iterable, Any]],
        tokenizer_encode: Callable[[str], list[int]],
        sample_rate: int,
        within_eos: bool,
    ):
        super().__init__()
        self.__transcribe = transcribe
        self.__tokenizer_encode = tokenizer_encode
        self.__SAMPLE_RATE = sample_rate
        self.__WITHIN_EOS = within_eos

    # override
    def _can_process(self, context: Context) -> ASRParam:
        return ASRParam.from_context(context)

    # override
    def _process(self, param: ASRParam) -> ASRResult:

        # 청크 합치기
        merged_vad_chunk = (
            param.vad_chunk
            if param.prev_vad_chunk.shape[0] == 0
            else np.concatenate([param.prev_vad_chunk, param.vad_chunk], axis=0)
        )

        # 추론
        segments, info = self.__transcribe(
            merged_vad_chunk, param.language, param.prompt
        )

        # Token 변환 및 시간 -> sample rate로 변경 및 오프셋 적용
        language = info.language or param.language
        candidate_tokens = []
        for segment in segments:
            tokens = [
                Token(
                    start=int(w.start * self.__SAMPLE_RATE) + param.prev_chunk_offset,
                    end=int(w.end * self.__SAMPLE_RATE) + param.prev_chunk_offset,
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
            candidate_tokens.extend(tokens)

        return ASRResult(
            merged_vad_chunk=merged_vad_chunk,
            candidate_tokens = candidate_tokens,
            language=info.language,
        )

    # override
    def _update(self, context: Context, result: ASRResult) -> None:
        result.update_context(context)
