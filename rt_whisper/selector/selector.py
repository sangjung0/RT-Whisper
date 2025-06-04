from rapidfuzz.distance import Levenshtein

from rt_whisper.abstracts import Worker
from rt_whisper.data import Token, Context

from .data import SelectorParam, SelectorResult


class Selector(Worker):
    def __init__(
        self,
        search_range_sc: int,
        threshold: float,
        padding: int,
        tolerance: int,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.__SEARCH_RANGE_SC = search_range_sc
        self.__THRESHOLD = threshold
        self.__PADDING = padding
        self.__TOLERANCE = tolerance
        self.__SMOOTH = smooth

    # override
    def _can_process(self, context: Context) -> SelectorParam:
        return SelectorParam.from_context(context)

    # override
    def _process(self, param: SelectorParam) -> SelectorResult:
        B = param.candidate_tokens
        A = param.prev_candidate_tokens
        language = param.language

        if not A:
            return SelectorResult(candidate_tokens=B)

        orphan_tokens = []
        new_token_group = []
        tail = []
        start = A[-1].start + self.__TOLERANCE[language]
        for t in B:
            if not t.is_word and t.end > start:
                tail.append(t)
            elif t.is_word and t.start > start:
                tail.append(t)
            else:
                new_token_group.append([t])

        idx_A, idx_group = -1, 0
        while True:
            idx_A += 1
            if idx_A >= len(A) or idx_group >= len(new_token_group):
                break

            a = A[idx_A]
            similarities = []
            for i in range(idx_group, len(new_token_group)):
                token = new_token_group[i][0]
                if token.start < a.start - self.__SEARCH_RANGE_SC[language]:
                    continue
                elif token.start > a.start + self.__SEARCH_RANGE_SC[language]:
                    break
                elif not token.is_word:
                    similarities.append((i, 0))
                else:
                    similarity = self.__token_similarity(
                        a, token, self.__PADDING[language]
                    )
                    similarities.append((i, similarity))

            if not similarities:
                orphan_tokens.append(a)
                continue

            max_arg = max(range(len(similarities)), key=lambda x: similarities[x][1])
            if similarities[max_arg][1] < self.__THRESHOLD[language]:
                orphan_tokens.append(a)
                continue
            idx = similarities[max_arg][0]
            new_token_group[idx].append(a)
            idx_group = similarities[0][0] + 1

        for i in range(idx_A, len(A)):
            orphan_tokens.append(A[i])

        tokens = [max(tk, key=lambda x: x.probability) for tk in new_token_group]
        tokens_idx = 0
        orphan_idx = 0
        while True:
            if orphan_idx >= len(orphan_tokens):
                break
            t = orphan_tokens[orphan_idx]
            if tokens_idx >= len(tokens):
                tokens.append(t)
                orphan_idx += 1
                continue
            token = tokens[tokens_idx]
            if token.is_word and token.start > t.start:
                tokens.insert(tokens_idx, t)
                orphan_idx += 1
            elif not token.is_word and token.end > t.end:
                tokens.insert(tokens_idx, t)
                orphan_idx += 1
            else:
                tokens_idx += 1
        tokens.extend(tail)

        return SelectorResult(candidate_tokens=tokens)

    # override
    def _update(self, context: Context, result: SelectorResult) -> None:
        result.update_context(context)

    def __token_iou(
        self, A: Token, B: Token, padding: int = 3200, smooth: float = 1e-6
    ) -> float:
        a1 = max(0, A.start - padding)
        b1 = A.end + padding
        a2 = max(0, B.start - padding)
        b2 = B.end + padding

        inner = max(0, min(b1, b2) - max(a1, a2))
        outer = max(max(b1, b2) - min(a1, a2), smooth)

        return inner / outer

    def __token_similarity(self, A: Token, B: Token, padding: int) -> float:
        # ratio = fuzz.ratio(A.text.strip().lower(), B.text.strip().lower())
        ratio = Levenshtein.normalized_similarity(A.tokens, B.tokens)
        iou = self.__token_iou(A, B, padding, self.__SMOOTH)
        return (ratio + iou) / 2
