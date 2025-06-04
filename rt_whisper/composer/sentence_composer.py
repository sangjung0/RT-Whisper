from rt_whisper.data import Context
from rt_whisper.tokenizers import Tokenizer

from . import Composer
from .data import ComposerParam, ComposerResult


class SentenceComposer(Composer):
    def __init__(self, max_prev_sent: int):
        super().__init__()
        self.__MAX_PREV_SENT = max_prev_sent + 1

    # override
    def _can_process(self, context: Context):
        if len(context.candidate_tokens) == 0:
            return None
        return ComposerParam(
            # 바로 빝에건 필요 없음
            completed_tokens=context.completed_tokens,
            candidate_tokens=context.candidate_tokens,
            language=context.language,
            order=context.order,
        )

    # override
    def _process(self, param: ComposerParam) -> ComposerResult:
        tokens = param.candidate_tokens
        tokenizer = Tokenizer.get_tokenizer(param.language or tokens[0].lang)

        if tokenizer is None:
            completed, _, completed_tokens, order = self._cut_by_eos(
                tokens, param.order, len(tokens)
            )
        else:
            completed, _, completed_tokens, order = self._cut_by_tokenizer(
                tokenizer, tokens, param.order, len(tokens)
            )

        candidate = []
        for idx in range(order - 1, max(order - 1 - self.__MAX_PREV_SENT, 0), -1):
            candidate.append(completed.pop())

        return ComposerResult(
            completed=completed,
            candidate=candidate,
            recycle_completed_tokens=completed_tokens,
            order=order - len(candidate),
        )
