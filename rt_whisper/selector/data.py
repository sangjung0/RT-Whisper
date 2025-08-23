from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

if TYPE_CHECKING:
    from rt_whisper.data import TokenState, Token


@dataclass(slots=True)
class SelectorContext:
    token_groups: list[list[Token]] = field(default_factory=list)


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
    prev_token_groups: list[list[Token]]

    @staticmethod
    def from_state(state: TokenState, sct_state: SelectorState) -> "SelectorParam":
        return SelectorParam(
            segment_tokens=state.segment_tokens,
            prev_token_groups=sct_state.prev.token_groups,
            language=state.language,
        )


@dataclass(slots=True)
class SelectorResult:
    segment_tokens: list[Token]
    token_groups: list[list[Token]]

    def update_state(self, state: TokenState, sct_state: SelectorState) -> None:
        state.segment_tokens = self.segment_tokens
        sct_state.context.token_groups = self.token_groups


@dataclass(slots=True)
class SelectorContextBuilderParam:
    anchor_timestamp: float
    token_groups: list[list[Token]]

    @staticmethod
    def from_state(
        state: TokenState, sct_state: SelectorState
    ) -> "SelectorContextBuilderParam":
        return SelectorContextBuilderParam(
            anchor_timestamp=state.anchor_timestamp,
            token_groups=sct_state.context.token_groups,
        )


@dataclass(slots=True)
class SelectorContextBuilderResult:
    context_token_groups: list[list[Token]]

    def update_state(self, state: SelectorState) -> None:
        state.context.token_groups = self.context_token_groups


__all__ = [
    "SelectorContext",
    "SelectorState",
    "SelectorParam",
    "SelectorResult",
    "SelectorContextBuilderParam",
    "SelectorContextBuilderResult",
]

