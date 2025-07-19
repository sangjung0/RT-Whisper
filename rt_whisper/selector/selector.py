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
        iou_threshold: float,
        cos_threshold: float,
        padding: int,
        logger: RTWhisperLogger,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.logger = logger

        self.__IOU_THRESHOLD = iou_threshold
        self.__COS_THRESHOLD = cos_threshold
        self.__PADDING = padding
        self.__SMOOTH = smooth

    # override
    def _can_process(self, context: TokenState) -> SelectorParam:
        sct_state: SelectorState = context.get_state(SelectorState)
        current = context.segment_tokens
        prev = sct_state.prev.segment_tokens
        if len(prev) == 0:
            return None
        elif len(current) >= len(prev) and all(
            c.text == p.text for c, p in zip(current[: len(prev)], prev)
        ):
            return None
        return SelectorParam.from_state(context, sct_state)

    # override
    def _process(self, param: SelectorParam) -> SelectorResult:
        self.logger.debug(f"Processing Selector", group_level=1)

        current = param.segment_tokens
        prev = param.prev_segment_tokens
        language = param.language
        self.logger.debug(
            f"Current: {''.join(str(t) for t in current if t.is_word)}", group_level=2
        )
        self.logger.debug(
            f"Previous: {''.join(str(t) for t in prev if t.is_word)}", group_level=2
        )

        token_groups = [[t] for t in prev]
        token_groups, orphan_tokens = group_similar_tokens(
            source=current,
            token_groups=token_groups,
            padding=self.__PADDING[language],
            iou_threshold=self.__IOU_THRESHOLD[language],
            cos_threshold=self.__COS_THRESHOLD[language],
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
            f"Context segment tokens: {''.join(str(t) for t in context_segment_tokens)}",
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
        iou_threshold: float,
        cos_threshold: float,
        padding: int,
        logger: RTWhisperLogger,
        smooth: float = 1e-6,
    ):
        super().__init__(
            iou_threshold=iou_threshold,
            cos_threshold=cos_threshold,
            padding=padding,
            logger=logger,
            smooth=smooth,
        )

    # override
    def _register_state(self, state: TokenState):
        state.set_state(SelectorState, SelectorState())
