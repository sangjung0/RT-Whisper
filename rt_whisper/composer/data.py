from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

if TYPE_CHECKING:
    from rt_whisper.data import TokenState, Token, Sentence


@dataclass(slots=True)
class ComposerContext:
    completed_tokens: list[Token] = field(default_factory=list)


@dataclass(slots=True)
class ComposerState:
    prev: ComposerContext = field(default_factory=ComposerContext)
    context: ComposerContext = field(default_factory=ComposerContext)

    def update(self, context: ComposerContext) -> None:
        assert isinstance(
            context, ComposerContext
        ), "context must be of type ComposerContext"

        self.__init__()
        self.prev = context

    def extract(self) -> ComposerContext:
        return self.context


@dataclass(slots=True)
class ComposerParam:
    segment_tokens: list[Token]
    order: int
    language: str | None
    completed_tokens: list[Token]
    prev_completed_tokens: list[Token]

    @staticmethod
    def from_context(state: TokenState, cps_state: ComposerState) -> "ComposerParam":
        return ComposerParam(
            completed_tokens=state.completed_tokens,
            order=state.order,
            language=state.language,
            segment_tokens=state.segment_tokens,
            prev_completed_tokens=cps_state.prev.completed_tokens,
        )


@dataclass(slots=True)
class ComposerResult:
    completed: list[Sentence]
    candidate: list[Sentence]
    order: int
    completed_tokens: list[Token]

    def update_context(self, state: TokenState, cps_state: ComposerState) -> None:
        state.completed = self.completed
        state.candidate = self.candidate
        state.order = self.order
        cps_state.context.completed_tokens = self.completed_tokens


__all__ = [
    "ComposerContext",
    "ComposerState",
    "ComposerParam",
    "ComposerResult",
]
