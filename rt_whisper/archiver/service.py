from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from rt_whisper.data import TokenState, Token
from rt_whisper.processors.asr.data import ASRState, ASRContext

if TYPE_CHECKING:
    pass


def state_to_dict(state: TokenState) -> dict:
    asr_state: ASRState = state.get_state(ASRState)
    return {
        "asr": {
            "prev_chunk": asr_state.prev.chunk.shape,
            "merged_chunk": asr_state.merged_chunk.shape,
        },
        "chunk_shape": state.chunk.shape,
        "offset": state.offset,
        "prompt": state.prompt,
        "language": state.language,
        "anchor_timestamp": state.anchor_timestamp,
        "segment_tokens": [token.to_dict() for token in state.segment_tokens],
    }


def dict_to_state(data: dict, state: TokenState) -> TokenState:
    asr_state = ASRState(
        merged_chunk=np.zeros(data["asr"]["merged_chunk"], dtype=np.float32),
        prev=ASRContext(chunk=np.zeros(data["asr"]["prev_chunk"], dtype=np.float32)),
    )
    state.set_state(ASRState, asr_state)
    state.chunk = np.zeros(data["chunk_shape"], dtype=np.float32)
    state.offset = data["offset"]
    state.prompt = data["prompt"]
    state.language = data["language"]
    state.anchor_timestamp = data["anchor_timestamp"]
    state.segment_tokens = [Token.from_dict(token) for token in data["segment_tokens"]]

    return state


__all__ = ["state_to_dict", "dict_to_state"]
