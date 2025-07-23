from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

if TYPE_CHECKING:
    from rt_whisper.data import TokenState, Token


@dataclass(slots=True)
class ProbabilityFilterState:
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    count: dict[str, int] = field(default_factory=dict)

    def update(self, state: "ProbabilityFilterState") -> None:
        self.mean.update(state.mean)
        self.std.update(state.std)
        self.count.update(state.count)

    def extract(self) -> "ProbabilityFilterState":
        return self


@dataclass(slots=True)
class ProbabilityFilterParam:
    segment_tokens: list[Token]
    language: str | None
    mean: float | None
    std: float | None
    count: float | None

    @staticmethod
    def from_context(
        state: TokenState, prob_state: ProbabilityFilterState
    ) -> "ProbabilityFilterParam":
        language = state.language
        return ProbabilityFilterParam(
            segment_tokens=state.segment_tokens,
            language=state.language,
            mean=prob_state.mean.get(language, None),
            std=prob_state.std.get(language, None),
            count=prob_state.count.get(language, None),
        )


@dataclass(slots=True)
class ProbabilityFilterResult:
    segment_tokens: list[Token]
    mean: float | None
    std: float | None
    count: float | None

    def update_context(
        self, state: TokenState, prob_state: ProbabilityFilterState
    ) -> None:
        language = state.language
        state.segment_tokens = self.segment_tokens
        prob_state.mean[language] = self.mean
        prob_state.std[language] = self.std
        prob_state.count[language] = self.count


@dataclass(slots=True)
class ProbabilityMinFilterParam:
    segment_tokens: float
    language: str | None

    @staticmethod
    def from_context(state: TokenState) -> "ProbabilityMinFilterParam":
        return ProbabilityMinFilterParam(
            segment_tokens=state.segment_tokens, language=state.language
        )


@dataclass(slots=True)
class ProbabilityMinFilterResult:
    segment_tokens: list[Token]

    def update_context(self, state: TokenState) -> None:
        state.segment_tokens = self.segment_tokens

__all__ = [
    "ProbabilityFilterState",
    "ProbabilityFilterParam",
    "ProbabilityFilterResult",
    "ProbabilityMinFilterParam",
    "ProbabilityMinFilterResult",
]
