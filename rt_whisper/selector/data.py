from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

if TYPE_CHECKING:
    from rt_whisper.data import TokenState, Token


@dataclass(slots=True)
class SelectorContext:
    segment_tokens: list[Token] = field(default_factory=list)


@dataclass(slots=True)
class SelectorState:
    prev: SelectorContext = field(default_factory=SelectorContext)
    context: SelectorContext = field(default_factory=SelectorContext)

    def update(self, context: SelectorContext) -> None:
        self.__init__()
        self.prev = context

    def extract(self) -> SelectorContext:
        return self.context


@dataclass(slots=True)
class SelectorParam:
    segment_tokens: list[Token]
    language: str | None
    prev_segment_tokens: list[Token]

    @staticmethod
    def from_state(state: TokenState, sct_state: SelectorState) -> "SelectorParam":
        return SelectorParam(
            segment_tokens=state.segment_tokens,
            prev_segment_tokens=sct_state.prev.segment_tokens,
            language=state.language,
        )


@dataclass(slots=True)
class SelectorResult:
    segment_tokens: list[Token]

    def update_state(self, state: TokenState) -> None:
        state.segment_tokens = self.segment_tokens


@dataclass(slots=True)
class SelectorContextBuilderParam:
    segment_tokens: list[Token]
    anchor_timestamp: float

    @staticmethod
    def from_state(state: TokenState) -> "SelectorContextBuilderParam":
        return SelectorContextBuilderParam(
            segment_tokens=state.segment_tokens,
            anchor_timestamp=state.anchor_timestamp,
        )


@dataclass(slots=True)
class SelectorContextBuilderResult:
    context_segment_tokens: list[Token]

    def update_state(self, state: SelectorState) -> None:
        state.context.segment_tokens = self.context_segment_tokens
