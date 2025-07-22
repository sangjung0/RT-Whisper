from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

from sj_utils.audio import generate_empty_chunk

from .result import Result

if TYPE_CHECKING:
    from typing import Type
    from .param import Param
    from .sentence import Sentence
    from .token import Token


@dataclass(slots=True)
class TokenState:
    # user defined fields
    chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    order: int = field(default=0)
    offset: int = field(default=0)
    prompt: str | None = field(default=None)
    language: str | None = field(default=None)

    # common fields
    anchor_timestamp: int = field(default=0)
    segment_tokens: list[Token] = field(default_factory=list)
    completed: list[Sentence] = field(default_factory=list)
    candidate: list[Sentence] = field(default_factory=list)

    state_dict: dict[Type, object] = field(default_factory=dict)

    def set_state(self, cls: Type, state: object):
        self.state_dict[cls] = state

    def get_state(self, cls: Type) -> object:
        return self.state_dict[cls]

    def bind(self, param: Param):
        self.chunk = param.chunk
        self.order = param.order
        self.offset = param.offset
        self.prompt = param.prompt
        self.language = param.language

        for cls, state in param.context_dict.items():
            self.state_dict[cls].update(state)

    def extract(self) -> Result:
        context_dict = {cls: state.extract() for cls, state in self.state_dict.items()}
        return Result(
            order=self.order,
            offset=self.offset,
            completed=self.completed,
            candidate=self.candidate,
            context_dict=context_dict,
        )
