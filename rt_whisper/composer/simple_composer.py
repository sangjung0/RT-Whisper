from rt_whisper.data import Context
from rt_whisper.tokenizers import Tokenizer

from .composer import Composer
from .data import ComposerParam, ComposerResult


class SimpleComposer(Composer):
    def __init__(self):
        super().__init__()

    # override
    def _can_process(self, context: Context) -> ComposerParam:
        if len(context.candidate_tokens) == 0:
            return False
        return ComposerParam(
            # 바로 밑에건 필요 없음
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
            completed, candidate, completed_tokens, order = self._cut_by_eos(
                tokens, param.order, len(tokens)
            )
        else:
            completed, candidate, completed_tokens, order = self._cut_by_tokenizer(
                tokenizer, param.tokens, order, len(tokens)
            )

        return ComposerResult(
            completed=completed,
            candidate=candidate,
            recycle_completed_tokens=completed_tokens,
            order=order,
        )
