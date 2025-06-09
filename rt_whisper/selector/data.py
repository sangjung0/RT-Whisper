from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext, Token


@dataclass(slots=True)
class SelectorParam:
    segment_tokens: list[Token]
    language: str | None
    prev_segment_tokens: list[Token]

    @staticmethod
    def from_context(context: TokenContext) -> "SelectorParam":
        return SelectorParam(
            segment_tokens=context.segment_tokens,
            prev_segment_tokens=context.prev_segment_tokens,
            language=context.language,
        )


@dataclass(slots=True)
class SelectorResult:
    segment_tokens: list[Token]

    def update_context(self, context: TokenContext) -> None:
        context.segment_tokens = self.segment_tokens
