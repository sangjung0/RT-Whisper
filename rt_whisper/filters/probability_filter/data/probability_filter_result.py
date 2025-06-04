from dataclasses import dataclass

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class ProbabilityFilterResult:
    candidate_tokens: list[Token]
    mean: float | None
    std: float | None
    count: float | None

    def update_context(self, context: Context) -> None:
        language = context.language
        context.candidate_tokens = self.candidate_tokens
        context.statistics["probability"]["mean"][language] = self.mean
        context.statistics["probability"]["std"][language] = self.std
        context.statistics["probability"]["count"][language] = self.count
