from dataclasses import dataclass

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class ComposerParam:
    completed_tokens: list[Token]
    candidate_tokens: list[Token]
    language: str | None
    order: int

    @staticmethod
    def validate(context: Context) -> bool:
        return len(context.completed_tokens) > 0 or len(context.candidate_tokens) > 0

    @staticmethod
    def from_context(context: Context) -> "ComposerParam":
        return ComposerParam(
            completed_tokens=context.completed_tokens,
            candidate_tokens=context.candidate_tokens,
            language=context.language,
            order=context.order,
        )
