from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from rt_whisper.data import TokenState, Token, Sentence

if TYPE_CHECKING:
    pass


def state_to_dict(state: TokenState) -> dict:
    return {
        "chunk": {
            "shape": state.chunk.shape,
        },
        "order": state.order,
        "offset": state.offset,
        "prompt": state.prompt,
        "language": state.language,
        "anchor_timestamp": state.anchor_timestamp,
        "segment_tokens": [token.to_dict() for token in state.segment_tokens],
        "completed": [sentence.to_dict() for sentence in state.completed],
        "candidate": [sentence.to_dict() for sentence in state.candidate],
    }


def dict_to_state(data: dict, state: TokenState) -> TokenState:
    state.chunk = np.zeros(data["chunk"]["shape"], dtype=np.float32)
    state.order = data["order"]
    state.offset = data["offset"]
    state.prompt = data["prompt"]
    state.language = data["language"]
    state.anchor_timestamp = data["anchor_timestamp"]
    state.segment_tokens = [Token.from_dict(token) for token in data["segment_tokens"]]
    state.completed = [Sentence.from_dict(sentence) for sentence in data["completed"]]
    state.candidate = [Sentence.from_dict(sentence) for sentence in data["candidate"]]

    return state
