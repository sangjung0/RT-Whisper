from dataclasses import dataclass

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class DurationFilterResult:
    candidate_tokens: list[Token]
    mean: float
    std: float
    count: float

    def update_context(self, context: Context):
        language = context.language
        context.candidate_tokens = self.candidate_tokens
        context.statistics["duration"]["mean"][language] = self.mean
        context.statistics["duration"]["std"][language] = self.std
        context.statistics["duration"]["count"][language] = self.count
