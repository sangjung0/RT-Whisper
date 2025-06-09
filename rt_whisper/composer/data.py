from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext, Token, Sentence


@dataclass(slots=True)
class ComposerParam:
    anchor_timestamp: int
    offset: int
    prev_completed_tokens: list[Token]
    segment_tokens: list[Token]
    language: str | None
    order: int

    @staticmethod
    def from_context(context: TokenContext) -> "ComposerParam":
        return ComposerParam(
            anchor_timestamp=context.anchor_timestamp,
            offset=context.offset,
            prev_completed_tokens=context.composer.prev.completed_tokens,
            segment_tokens=context.segment_tokens,
            language=context.language,
            order=context.order,
        )


@dataclass(slots=True)
class ComposerResult:
    completed: list[Sentence]
    candidate: list[Sentence]
    recycle_segment_tokens: list[Token]
    recycle_completed_tokens: list[Token]
    order: int

    def update_context(self, context: TokenContext):
        context.completed = self.completed
        context.candidate = self.candidate
        context.recycle_segment_tokens = self.recycle_segment_tokens
        context.composer.recycle.completed_tokens = self.recycle_completed_tokens
        context.order = self.order


@dataclass(slots=True)
class ComposerRecycle:
    completed_tokens: list[Token] = field(default_factory=list)


@dataclass(slots=True)
class ComposerStorage:
    recycle: ComposerRecycle = field(default_factory=ComposerRecycle)
    prev: ComposerRecycle = field(default_factory=ComposerRecycle)

    def update(self, recycle: ComposerRecycle) -> None:
        self.__init__()
        if recycle is not None:
            self.prev = recycle

    def extract(self) -> ComposerRecycle:
        return self.recycle
