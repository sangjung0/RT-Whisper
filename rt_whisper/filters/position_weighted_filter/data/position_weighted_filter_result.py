from dataclasses import dataclass

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class PositionWeightedFilterResult:
    # tokens: list[Token]

    def update_context(self, context:Context) -> None:
        # context.merged_candidate_tokens = self.tokens
        pass
