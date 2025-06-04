from dataclasses import dataclass
from typing import Union

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class DurationFilterParam:
    candidate_tokens: list[Token]
    language: Union[str, None]
    mean: Union[float, None]
    std: Union[float, None]
    count: Union[int, None]

    @staticmethod
    def validate(context: Context):
        return context.merged_vad_chunk_size > 0 and len(context.candidate_tokens) > 0

    @staticmethod
    def from_context(context: Context) -> "DurationFilterParam":
        if DurationFilterParam.validate(context):
            language = context.language
            return DurationFilterParam(
                candidate_tokens=context.candidate_tokens,
                language=language,
                mean=context.statistics["duration"]["mean"].get(language, None),
                std=context.statistics["duration"]["std"].get(language, None),
                count=context.statistics["duration"]["count"].get(language, None),
            )
        return None
