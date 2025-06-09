from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext, Token


@dataclass(slots=True)
class ProbabilityFilterParam:
    segment_tokens: list[Token]
    language: str | None
    mean: float | None
    std: float | None
    count: float | None

    @staticmethod
    def from_context(context: TokenContext) -> "ProbabilityFilterParam":
        language = context.language
        return ProbabilityFilterParam(
            segment_tokens=context.segment_tokens,
            language=context.language,
            mean=context.probability_filter.mean.get(language, None),
            std=context.probability_filter.std.get(language, None),
            count=context.probability_filter.count.get(language, None),
        )


@dataclass(slots=True)
class ProbabilityFilterResult:
    segment_tokens: list[Token]
    mean: float | None
    std: float | None
    count: float | None

    def update_context(self, context: TokenContext) -> None:
        language = context.language
        context.segment_tokens = self.segment_tokens
        context.probability_filter.mean[language] = self.mean
        context.probability_filter.std[language] = self.std
        context.probability_filter.count[language] = self.count


@dataclass(slots=True)
class ProbabilityFilterStorage:
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    count: dict[str, int] = field(default_factory=dict)

    def update(self, recycle: "ProbabilityFilterStorage") -> None:
        if recycle is None:
            return
        self.mean.update(recycle.mean)
        self.std.update(recycle.std)
        self.count.update(recycle.count)

    def extract(self) -> "ProbabilityFilterStorage":
        return self
