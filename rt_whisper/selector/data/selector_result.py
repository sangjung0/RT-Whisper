from dataclasses import dataclass

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class SelectorResult:
    candidate_tokens: list[Token]

    def update_context(self, context: Context) -> None:
        context.candidate_tokens = self.candidate_tokens
