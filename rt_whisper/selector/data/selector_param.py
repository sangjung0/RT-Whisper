from dataclasses import dataclass

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class SelectorParam:
    candidate_tokens: list[Token]
    prev_candidate_tokens: list[Token]
    language: str | None

    @staticmethod
    def validate(context: Context) -> bool:
        return len(context.prev_candidate_tokens) > 0

    @staticmethod
    def from_context(context: Context) -> "SelectorParam":
        if SelectorParam.validate(context):
            return SelectorParam(
                candidate_tokens=context.candidate_tokens,
                prev_candidate_tokens=context.prev_candidate_tokens,
                language=context.language,
            )
        return None
