from dataclasses import dataclass
from typing import Union

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class ProbabilityFilterParam:
    candidate_tokens: list[Token]
    language: Union[str, None]
    mean: Union[float, None]
    std: Union[float, None]
    count: Union[float, None]

    @staticmethod
    def validate(context: Context) -> bool:
        return context.merged_vad_chunk_size > 0 and len(context.candidate_tokens) > 0

    @staticmethod
    def from_context(context: Context) -> "ProbabilityFilterParam":
        if ProbabilityFilterParam.validate(context):
            return ProbabilityFilterParam(
                candidate_tokens=context.candidate_tokens,
                language=context.language,
                mean=context.statistics["probability"]["mean"].get(
                    context.language, None
                ),
                std=context.statistics["probability"]["std"].get(
                    context.language, None
                ),
                count=context.statistics["probability"]["count"].get(
                    context.language, None
                ),
            )
        return None
