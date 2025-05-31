from typing import Generator
from whisper import tokenizer
from faster_whisper.transcribe import Segment

from settings import Settings
from abstracts import Pipeline
from data import Context, Token


class SegmentsToTokenWithEOS(Pipeline):
    def __init__(
        self,
        tokenizer: tokenizer.Tokenizer,
        SAMPLE_RATE: int = Settings.MODEL_SAMPLE_RATE,
    ):
        super().__init__()
        self.__tokenizer = tokenizer
        self._SAMPLE_RATE = SAMPLE_RATE

    def can_process(self, context: Context) -> bool:
        if not context.merged_candidate_tokens:
            return False
        return context.merged_candidate_tokens, context.language

    def compute_process(self, param: tuple):
        segments: Generator[Segment] = param[0]
        language = param[1]

        new_tokens = []
        for segment in segments:
            tokens = [
                Token(
                    start=int(w.start * self._SAMPLE_RATE),
                    end=int(w.end * self._SAMPLE_RATE),
                    text=w.word,
                    lang=language,
                    tokens=self.__tokenizer.encode(w.word.lower()),
                    probability=w.probability,
                )
                for w in segment.words
            ]
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
            new_tokens.extend(tokens)

        return new_tokens

    def apply_process(self, context: Context, result) -> None:
        context.merged_candidate_tokens = result
