from __future__ import annotations
from typing import TYPE_CHECKING

from rt_whisper.abstracts import Worker
from rt_whisper.selector.data import (
    SelectorParam,
    SelectorResult,
    SelectorState,
    SelectorContextBuilderParam,
    SelectorContextBuilderResult,
)
from rt_whisper.selector.service import (
    position_tokens,
    group_similar_tokens,
    merge_tokens,
)

if TYPE_CHECKING:
    from rt_whisper import RTWhisperLogger
    from rt_whisper.data import TokenState

N = "\n\t\t\t"


class SelectorProcessor(Worker):
    def __init__(
        self,
        search_range_sc: int,
        threshold: float,
        padding: int,
        tolerance: int,
        logger: RTWhisperLogger,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.logger = logger

        self.__SEARCH_RANGE_SC = search_range_sc
        self.__THRESHOLD = threshold
        self.__PADDING = padding
        self.__TOLERANCE = tolerance
        self.__SMOOTH = smooth

    # override
    def _can_process(self, context: TokenState) -> SelectorParam:
        sct_state: SelectorState = context.get_state(SelectorState)
        if len(context.segment_tokens) > 0 or len(sct_state.prev.segment_tokens) > 0:
            return SelectorParam.from_state(context, sct_state)
        return None

    # override
    def _process(self, param: SelectorParam) -> SelectorResult:
        self.logger.debug(f"Processing Selector", group_level=1)

        B = param.segment_tokens
        A = param.prev_segment_tokens
        language = param.language
        self.logger.debug(
            f"Current: {''.join(str(t) for t in B if t.is_word)}", group_level=2
        )
        self.logger.debug(
            f"Previous: {''.join(str(t) for t in A if t.is_word)}", group_level=2
        )

        if not A:
            return SelectorResult(segment_tokens=B)

        token_groups, rest = position_tokens(
            source=A,
            target=B,
            tolerance=self.__TOLERANCE[language],
        )
        self.logger.debug(
            f"Token groups: {N}{N.join(', '.join(str(t) for t in g) for g in token_groups)}",
            group_level=2,
        )
        self.logger.debug(
            f"Rest tokens: {''.join(str(t) for t in rest if t.is_word)}", group_level=2
        )

        token_groups, orphan_tokens = group_similar_tokens(
            source=A,
            token_groups=token_groups,
            search_range=self.__SEARCH_RANGE_SC[language],
            padding=self.__PADDING[language],
            threshold=self.__THRESHOLD[language],
            smooth=self.__SMOOTH,
        )
        self.logger.debug(
            f"Grouped tokens: {N}{N.join(', '.join(str(t) for t in g) for g in token_groups)}",
            group_level=2,
        )
        self.logger.debug(
            f"Orphan tokens: {''.join(str(t) for t in orphan_tokens if t.is_word)}",
            group_level=2,
        )

        new_tokens = merge_tokens(
            token_groups=token_groups,
            orphan_tokens=orphan_tokens,
            rest=rest,
        )
        self.logger.debug(
            f"New tokens: {''.join(str(t) for t in new_tokens if t.is_word)}",
            group_level=2,
        )

        return SelectorResult(segment_tokens=new_tokens)

    # override
    def _update(self, context: TokenState, result: SelectorResult) -> None:
        result.update_state(context)


class SelectorContextBuilder(SelectorProcessor):
    # override
    def _can_build(self, state: TokenState) -> SelectorParam:
        return SelectorContextBuilderParam.from_state(state)

    # override
    def _context_build(self, param: SelectorContextBuilderParam):
        self.logger.debug(f"Building SelectorContext", group_level=1)

        context_segment_tokens = [
            t
            for t in param.segment_tokens
            if t.is_word and t.end > param.anchor_timestamp
        ]
        self.logger.debug(
            f"Context segment tokens: {', '.join(str(t) for t in context_segment_tokens)}",
            group_level=2,
        )

        return SelectorContextBuilderResult(
            context_segment_tokens=context_segment_tokens
        )

    # override
    def _context_update(
        self, state: TokenState, result: SelectorContextBuilderResult
    ) -> None:
        sct_state = state.get_state(SelectorState)
        result.update_state(sct_state)


class Selector(SelectorContextBuilder):
    def __init__(
        self,
        search_range_sc: int,
        threshold: float,
        padding: int,
        tolerance: int,
        logger: RTWhisperLogger,
        smooth: float = 1e-6,
    ):
        super().__init__(
            search_range_sc=search_range_sc,
            threshold=threshold,
            padding=padding,
            tolerance=tolerance,
            logger=logger,
            smooth=smooth,
        )

    # override
    def _register_state(self, state: TokenState):
        state.set_state(SelectorState, SelectorState())
