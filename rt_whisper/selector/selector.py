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
from rt_whisper.composer.data import ComposerState
from rt_whisper.selector.service import (
    group_similar_tokens,
    select_tokens,
    new_group_tokens,
    completed_and_candidate_tokens,
    filter_token_groups,
)

if TYPE_CHECKING:
    from rt_whisper.rt_whisper_logger import RTWhisperLogger
    from rt_whisper.data import TokenState
    from sj_utils.collection import SafetyDict

N = "\n\t\t\t"


class SelectorProcessor(Worker):
    def __init__(
        self,
        iou_threshold: SafetyDict[str, float],
        cos_threshold: SafetyDict[str, float],
        padding: SafetyDict[str, int],
        logger: RTWhisperLogger,
        m: float = 1,
        p: float = 1,
        s: float = 1,
        smooth: float = 1e-6,
    ):
        if m < 0 or p < 0 or s < 0 or m + p > 1 or s > 1:
            raise ValueError("Invalid weights for selection criteria")

        super().__init__()
        self.logger = logger

        self.__IOU_THRESHOLD = iou_threshold
        self.__COS_THRESHOLD = cos_threshold
        self.__PADDING = padding
        self.__SMOOTH = smooth

        self.__M = m
        self.__P = p
        self.__C = 1 - m - p

        self.__S = s
        self.__I = 1 - s

    # override
    def _can_process(self, context: TokenState) -> SelectorParam:
        sct_state: SelectorState = context.get_state(SelectorState)
        return SelectorParam.from_state(context, sct_state)

    # override
    def _process(self, param: SelectorParam) -> SelectorResult:
        self.logger.debug(f"Processing Selector", group_level=1)

        current = param.segment_tokens
        token_groups = param.prev_token_groups
        language = param.language
        self.logger.debug(
            f"Current: {''.join(str(t) for t in current if t.is_word)}", group_level=2
        )
        self.logger.debug(
            f"Previous Grouped tokens: {N}{N.join(', '.join(str(t) for t in g) for g in token_groups)}",
            group_level=2,
        )

        token_groups, orphan_tokens = group_similar_tokens(
            source=current,
            token_groups=token_groups,
            padding=self.__PADDING[language],
            iou_threshold=self.__IOU_THRESHOLD[language],
            cos_threshold=self.__COS_THRESHOLD[language],
            smooth=self.__SMOOTH,
            i=self.__I,
            s=self.__S,
        )
        self.logger.debug(
            f"Grouped tokens: {N}{N.join(', '.join(str(t) for t in g) for g in token_groups)}",
            group_level=2,
        )
        self.logger.debug(
            f"Orphan tokens: {''.join(str(t) for t in orphan_tokens if t.is_word)}",
            group_level=2,
        )

        token_groups = new_group_tokens(token_groups, orphan_tokens)
        self.logger.debug(
            f"Sorted Grouped tokens: {N}{N.join(', '.join(str(t) for t in g) for g in token_groups)}",
            group_level=2,
        )

        new_tokens = select_tokens(
            token_groups=token_groups, m=self.__M, p=self.__P, c=self.__C
        )
        self.logger.debug(
            f"New tokens: {''.join(str(t) for t in new_tokens if t.is_word)}",
            group_level=2,
        )

        completed_token, candidate_token = completed_and_candidate_tokens(
            new_tokens, param.anchor_timestamp
        )

        return SelectorResult(
            segment_tokens=new_tokens,
            token_groups=token_groups,
            completed_tokens=completed_token,
            candidate_tokens=candidate_token,
        )

    # override
    def _update(self, context: TokenState, result: SelectorResult) -> None:
        sct_state: SelectorState = context.get_state(SelectorState)
        result.update_state(context, sct_state)


class SelectorContextBuilder(SelectorProcessor):
    def __init__(
        self,
        *args,
        token_group_size: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__TOKEN_GROUP_SIZE = token_group_size

    # override
    def _can_build(self, state: TokenState) -> SelectorParam:
        sct_state: SelectorState = state.get_state(SelectorState)
        return SelectorContextBuilderParam.from_state(state, sct_state)

    # override
    def _context_build(self, param: SelectorContextBuilderParam):
        self.logger.debug(f"Building SelectorContext", group_level=1)
        token_groups = filter_token_groups(
            param.token_groups, len(param.completed_tokens), self.__TOKEN_GROUP_SIZE
        )
        self.logger.debug(
            f"Context segment tokens: {N}{N.join(', '.join(str(t) for t in g) for g in token_groups)}",
            group_level=2,
        )
        return SelectorContextBuilderResult(context_token_groups=token_groups)

    # override
    def _context_update(
        self, state: TokenState, result: SelectorContextBuilderResult
    ) -> None:
        sct_state = state.get_state(SelectorState)
        result.update_state(sct_state)


class Selector(SelectorContextBuilder):
    def __init__(
        self,
        iou_threshold: SafetyDict[str, float],
        cos_threshold: SafetyDict[str, float],
        padding: SafetyDict[str, int],
        logger: RTWhisperLogger,
        token_group_size: int,
        m: float = 1,
        p: float = 1,
        s: float = 1,
        smooth: float = 1e-6,
    ):
        super().__init__(
            iou_threshold=iou_threshold,
            cos_threshold=cos_threshold,
            padding=padding,
            logger=logger,
            m=m,
            p=p,
            s=s,
            token_group_size=token_group_size,
            smooth=smooth,
        )

    # override
    def _register_state(self, state: TokenState):
        state.set_state(SelectorState, SelectorState())


__all__ = ["Selector"]
