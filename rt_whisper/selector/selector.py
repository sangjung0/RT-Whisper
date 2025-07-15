from __future__ import annotations
from typing import TYPE_CHECKING

from logging import Logger

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
    from rt_whisper.data import TokenState


class SelectorProcessor(Worker):
    def __init__(
        self,
        search_range_sc: int,
        threshold: float,
        padding: int,
        tolerance: int,
        logger: Logger,
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
        self.logger.debug(f"\tProcessing Selector")

        B = param.segment_tokens
        A = param.prev_segment_tokens
        language = param.language
        self.logger.debug(f"\t\tCurrent: {B}")
        self.logger.debug(f"\t\tPrevious: {A}")

        if not A:
            return SelectorResult(segment_tokens=B)

        token_groups, rest = position_tokens(
            source=A,
            target=B,
            tolerance=self.__TOLERANCE[language],
        )
        self.logger.debug(f"\t\tToken groups: {token_groups}")
        self.logger.debug(f"\t\tRest tokens: {rest}")

        token_groups, orphan_tokens = group_similar_tokens(
            source=A,
            token_groups=token_groups,
            search_range=self.__SEARCH_RANGE_SC[language],
            padding=self.__PADDING[language],
            threshold=self.__THRESHOLD[language],
            smooth=self.__SMOOTH,
        )
        self.logger.debug(f"\t\tGrouped tokens: {token_groups}")
        self.logger.debug(f"\t\tOrphan tokens: {orphan_tokens}")

        new_tokens = merge_tokens(
            token_groups=token_groups,
            orphan_tokens=orphan_tokens,
            rest=rest,
        )
        self.logger.debug(f"\t\tNew tokens: {new_tokens}")

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
        self.logger.debug(f"\tBuilding SelectorContext")

        context_segment_tokens = [
            t
            for t in param.segment_tokens
            if t.is_word and t.end > param.anchor_timestamp
        ]
        self.logger.debug(f"\t\tContext segment tokens: {context_segment_tokens}")

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
        logger: Logger,
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
