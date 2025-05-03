from RTWhisper.data import Context
from RTWhisper import Tokenizer

from .Composer import Composer


class SentenceComposer(Composer):
    def __init__(self, max_prev_sent: int):
        super().__init__()
        self.__MAX_PREV_SENT = max_prev_sent + 1

    def can_process(self, context: Context):
        if len(context.merged_candidate_tokens) == 0:
            return False
        return context.merged_candidate_tokens, context.language, context.order

    def compute_process(self, param):
        (tokens, language, order) = param

        language = language if language else tokens[0].lang
        tokenizer = Tokenizer.get_tokenizer(language)

        if tokenizer is None:
            completed, _, completed_tokens, order = self._cut_by_eos(
                tokens, order, len(tokens)
            )
        else:
            completed, _, completed_tokens, order = self._cut_by_tokenizer(
                tokenizer, tokens, order, len(tokens)
            )

        candidate = []
        for idx in range(order - 1, order - 1 - self.__MAX_PREV_SENT, -1):
            if idx < 0:
                break
            candidate.append(completed.pop())

        return completed, candidate, completed_tokens, order - len(candidate)
