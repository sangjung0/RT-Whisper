from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from typing import Type
    from .sentence import Sentence


@dataclass(slots=True, frozen=True)
class Result:
    order: int
    offset: int

    completed: list[Sentence]
    candidate: list[Sentence]

    context_dict: dict[Type, object]
