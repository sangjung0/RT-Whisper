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
    anchor_timestamp: int
    prev_completed_tokens: list[Token]

    @staticmethod
    def from_context(state: TokenState, cps_state: ComposerState) -> "ComposerParam":
        return ComposerParam(
            anchor_timestamp=state.anchor_timestamp,
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

    def update_context(self, state: TokenState):
        state.completed = self.completed
        state.candidate = self.candidate
        state.order = self.order


@dataclass(slots=True)
class ComposerContextBuilderParam:
    candidate: list[Sentence]
    anchor_timestamp: int

    @staticmethod
    def from_state(state: TokenState) -> "ComposerContextBuilderParam":
        return ComposerContextBuilderParam(
            candidate=state.candidate,
            anchor_timestamp=state.anchor_timestamp,
        )


@dataclass(slots=True)
class ComposerContextBuilderResult:
    context_completed_tokens: list[Token]

    def update_context(self, cps_state: ComposerState):
        cps_state.context.completed_tokens = self.context_completed_tokens


__all__ = [
    "ComposerContext",
    "ComposerState",
    "ComposerParam",
    "ComposerResult",
    "ComposerContextBuilderParam",
    "ComposerContextBuilderResult",
]
