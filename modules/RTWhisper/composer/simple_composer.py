from RTWhisper.data import Context
from RTWhisper import Tokenizer

from .composer_ import Composer


class SimpleComposer(Composer):
    def __init__(self):
        super().__init__()

    def can_process(self, context: Context) -> bool:
        if len(context.merged_candidate_tokens) == 0:
            return False
        return context.merged_candidate_tokens, context.language, context.order

    def compute_process(self, param):
        (tokens, language, order) = param

        language = language if language else tokens[0].lang
        tokenizer = Tokenizer.get_tokenizer(language)

        if tokenizer is None:
            completed, candidate, completed_tokens, order = self._cut_by_eos(
                tokens, order, len(tokens)
            )
        else:
            completed, candidate, completed_tokens, order = self._cut_by_tokenizer(
                tokenizer, tokens, order, len(tokens)
            )

        return completed, candidate, completed_tokens, order
