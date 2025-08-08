from __future__ import annotations
from typing import TYPE_CHECKING

from rt_whisper.abstracts import Worker
from rt_whisper.composer.data import ComposerState, ComposerParam, ComposerResult
from rt_whisper.composer.service import cut_by_eos, tokens_to_sentences

if TYPE_CHECKING:
    from rt_whisper.data import TokenState


class SimpleComposer(Worker):

    # override
    def _register_state(self, state: TokenState):
        state.set_state(ComposerState, ComposerState())

    # override
    def _can_process(self, state: TokenState):
        cps_state: ComposerState = state.get_state(ComposerState)
        if len(cps_state.prev.completed_tokens) > 0 or len(state.segment_tokens) > 0:
            return ComposerParam.from_context(state, cps_state)
        return None

    # override
    def _process(self, param: ComposerParam) -> ComposerResult:
        segments = cut_by_eos(param.segment_tokens)
        sentences, order = tokens_to_sentences(segments, param.order)

        return ComposerResult(
            completed=sentences,
            candidate=[],
            order=order,
        )

    # override
    def _update(self, state: TokenState, result: ComposerResult):
        result.update_context(state)


__all__ = ["SimpleComposer"]
