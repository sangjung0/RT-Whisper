from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from rt_whisper.data import TokenState, Token

if TYPE_CHECKING:
    pass


def state_to_dict(state: TokenState) -> dict:
    return {
        "chunk_shape": state.chunk.shape,
        "offset": state.offset,
        "prompt": state.prompt,
        "language": state.language,
        "anchor_timestamp": state.anchor_timestamp,
        "segment_tokens": [token.to_dict() for token in state.segment_tokens],
    }


def dict_to_state(data: dict, state: TokenState) -> TokenState:
    state.chunk = np.zeros(data["chunk_shape"], dtype=np.float32)
    state.offset = data["offset"]
    state.prompt = data["prompt"]
    state.language = data["language"]
    state.anchor_timestamp = data["anchor_timestamp"]
    state.segment_tokens = [Token.from_dict(token) for token in data["segment_tokens"]]

    return state
