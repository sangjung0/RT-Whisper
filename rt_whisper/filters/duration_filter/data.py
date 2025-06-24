from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from rt_whisper.data import TokenState, Token


@dataclass(slots=True)
class DurationFilterState:
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    count: dict[str, int] = field(default_factory=dict)

    def update(self, state: "DurationFilterState"):
        self.mean.update(state.mean)
        self.std.update(state.std)
        self.count.update(state.count)

    def extract(self) -> "DurationFilterState":
        return self


@dataclass(slots=True)
class DurationFilterParam:
    segment_tokens: list[Token]
    language: str | None
    mean: float | None
    std: float | None
    count: int | None

    @staticmethod
    def from_state(
        state: TokenState, dfs_state: DurationFilterState
    ) -> "DurationFilterParam":
        language = state.language
        return DurationFilterParam(
            segment_tokens=state.segment_tokens,
            language=language,
            mean=dfs_state.mean.get(language, None),
            std=dfs_state.std.get(language, None),
            count=dfs_state.count.get(language, None),
        )


@dataclass(slots=True)
class DurationFilterResult:
    segment_tokens: list[Token]
    mean: float
    std: float
    count: float

    def update_state(self, state: TokenState, dfs_state: DurationFilterState):
        language = state.language
        state.segment_tokens = self.segment_tokens
        dfs_state.mean[language] = self.mean
        dfs_state.std[language] = self.std
        dfs_state.count[language] = self.count
