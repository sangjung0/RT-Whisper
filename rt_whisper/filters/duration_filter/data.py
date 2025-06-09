from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext, Token


@dataclass(slots=True)
class DurationFilterParam:
    segment_tokens: list[Token]
    language: str | None
    mean: float | None
    std: float | None
    count: int | None

    @staticmethod
    def from_context(context: TokenContext) -> "DurationFilterParam":
        language = context.language
        return DurationFilterParam(
            segment_tokens=context.segment_tokens,
            language=language,
            mean=context.duration_filter.mean.get(language, None),
            std=context.duration_filter.std.get(language, None),
            count=context.duration_filter.count.get(language, None),
        )


@dataclass(slots=True)
class DurationFilterResult:
    segment_tokens: list[Token]
    mean: float
    std: float
    count: float

    def update_context(self, context: TokenContext):
        language = context.language
        context.segment_tokens = self.segment_tokens
        context.duration_filter.mean[language] = self.mean
        context.duration_filter.std[language] = self.std
        context.duration_filter.count[language] = self.count


@dataclass(slots=True)
class DurationFilterStorage:
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    count: dict[str, int] = field(default_factory=dict)

    def update(self, recycle: "DurationFilterStorage"):
        if recycle is None:
            return
        self.mean.update(recycle.mean)
        self.std.update(recycle.std)
        self.count.update(recycle.count)

    def extract(self) -> "DurationFilterStorage":
        return self
