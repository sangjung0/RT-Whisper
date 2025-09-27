from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from typing import Type
    from rt_whisper.data.sentence import Sentence
    from rt_whisper.data.token import Token


@dataclass(slots=True, frozen=True)
class Result:
    order: int
    offset: int

    completed: list[Sentence]
    candidate: list[Sentence]
    completed_tokens: list[Token]
    candidate_tokens: list[Token]

    context_dict: dict[Type, object]


__all__ = ["Result"]
