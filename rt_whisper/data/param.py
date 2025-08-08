from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from typing import Type
from dataclasses import dataclass, field

from sj_utils.audio import generate_empty_chunk

if TYPE_CHECKING:
    from rt_whisper.data.result import Result


@dataclass(slots=True)
class Param:
    chunk: np.ndarray = field(default_factory=generate_empty_chunk)
    order: int = field(default=0)
    offset: int = field(default=0)
    prompt: str | None = field(default=None)
    language: str | None = field(default=None)

    context_dict: dict[Type, object] = field(default_factory=dict)

    def validate_audio(self, v: np.ndarray):
        return isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype == np.float32

    def __post_init__(self):
        if not self.validate_audio(self.chunk):
            raise ValueError("chunk must be a 1D numpy array of float32")

    def update(self, result: Result, update_prompt: bool = True):
        self.__init__()

        self.order = result.order
        self.offset = result.offset
        self.context_dict = result.context_dict

        if update_prompt:
            from rt_whisper.composer.data import ComposerState

            if ComposerState in self.context_dict:
                completed_token = self.context_dict[ComposerState].completed_tokens
                completed = result.completed
                prompt = completed[-1].text if completed else ""
                prompt += "".join(
                    token.text for token in completed_token if token.is_word
                )
                self.prompt = prompt


__all__ = ["Param"]
