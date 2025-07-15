from __future__ import annotations
from typing import TYPE_CHECKING

from logging import Logger

from rt_whisper.abstracts import Worker
from rt_whisper.tokenizers import Tokenizer
from rt_whisper.composer.data import (
    ComposerParam,
    ComposerResult,
    ComposerContextBuilderParam,
    ComposerContextBuilderResult,
    ComposerState,
)
from rt_whisper.composer.service import (
    select_language,
    cut_by_eos,
    cut_by_tokenizer,
    tokens_to_sentences,
    classify_candidate_completed,
    context_tokens,
)


if TYPE_CHECKING:
    from rt_whisper.data import TokenState


class Composer(Worker):

    def __init__(self, logger: Logger):
        super().__init__()
        self.logger = logger

    # override
    def _register_state(self, state: TokenState):
        state.set_state(ComposerState, ComposerState())

    # override
    def _can_process(self, state: TokenState) -> ComposerParam:
        cps_state: ComposerState = state.get_state(ComposerState)
        if len(cps_state.prev.completed_tokens) > 0 or len(state.segment_tokens) > 0:
            return ComposerParam.from_context(state, cps_state)
        return None

    # override
    def _process(self, param: ComposerParam) -> ComposerResult:
        self.logger.debug(f"\tProcessing composer")

        tokens = param.prev_completed_tokens + param.segment_tokens
        self.logger.debug(f"\t\tTokens: {len(tokens)}")

        language = select_language(param.language, tokens)
        self.logger.debug(f"\t\tLanguage: {language}")

        tokenizer = Tokenizer.get_tokenizer(language)

        if tokenizer is None:
            segments = cut_by_eos(tokens)
        else:
            segments = cut_by_tokenizer(tokenizer, tokens, self.logger)
        self.logger.debug(f"\t\tSegments: {segments}")

        sentences, order = tokens_to_sentences(segments, param.order)
        completed, candidate = classify_candidate_completed(
            sentences, param.anchor_timestamp
        )
        self.logger.debug(f"\t\tCompleted: {completed}")
        self.logger.debug(f"\t\tCandidate: {candidate}")

        order -= len(candidate)

        return ComposerResult(
            completed=completed,
            candidate=candidate,
            order=order,
        )

    # override
    def _update(self, state: TokenState, result: ComposerResult):
        result.update_context(state)

    # override
    def _can_build(self, state: TokenState):
        return ComposerContextBuilderParam.from_state(state)

    # override
    def _context_build(self, param: ComposerContextBuilderParam):
        self.logger.debug(f"\tBuilding context for composer")
        completed_tokens = context_tokens(param.candidate, param.anchor_timestamp)
        self.logger.debug(f"\t\tCompleted tokens: {completed_tokens}")
        return ComposerContextBuilderResult(
            context_completed_tokens=completed_tokens,
        )

    # override
    def _context_update(self, state: TokenState, result: ComposerContextBuilderResult):
        cps_state: ComposerState = state.get_state(ComposerState)
        result.update_context(cps_state)
