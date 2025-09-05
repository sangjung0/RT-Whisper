from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from typing import Callable
from functools import lru_cache, reduce
from operator import mul, add

if TYPE_CHECKING:
    from rt_whisper.data import Token


def __cosine_similarity(A: Token, B: Token) -> float:
    if A.text == B.text:
        return 1
    if hash(A) < hash(B):
        A, B = B, A
    return __lru_cosine_similarity(A, B)


@lru_cache(maxsize=4096)
def __lru_cosine_similarity(A: Token, B: Token) -> float:
    return torch.nn.functional.cosine_similarity(A.embedding, B.embedding, dim=0).item()


def __token_iou(A: Token, B: Token, padding: int, smooth: float = 1e-6) -> float:
    a1 = max(0, A.start - padding)
    b1 = A.end + padding
    a2 = max(0, B.start - padding)
    b2 = B.end + padding

    inner = max(0, min(b1, b2) - max(a1, a2))
    outer = max(max(b1, b2) - min(a1, a2), smooth)

    return inner / outer


def select_best_only_confidence(group: list[Token], _, __, ___, ____) -> Token:
    tokens = [t for t in group if t.is_word]
    if not tokens:
        return group[0]
    return max(tokens, key=lambda t: t.probability)


def select_best_only_prev(
    group: list[Token], prev: torch.Tensor | None, _, __, ___
) -> Token:
    tokens = [t for t in group if t.is_word]
    if not tokens:
        return group[0]
    elif len(set(t.text for t in tokens)) == 1:
        return max(tokens, key=lambda t: t.probability)

    similarities = []
    # print(f"Selecting best token from group of {len(tokens)} tokens")
    for t in tokens:
        # print(f"\tToken: {t.text}")
        sim = (
            1
            if prev is None
            else torch.nn.functional.cosine_similarity(t.embedding, prev, dim=0).item()
        )
        sim = (sim + 1) / 2  # Normalize to [0, 1]

        # print(
        #     f"\t\tMean similarity: {mean_sim}, Previous similarity: {prev_sim},  Combined: {sim}"
        # )
        similarities.append(sim)

    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    return tokens[best_idx]


def select_best_confidence_and_prev(
    group: list[Token], prev: torch.Tensor | None, _, p: float = 1, c: float = 1
) -> Token:
    tokens = [t for t in group if t.is_word]
    if not tokens:
        return group[0]
    elif len(set(t.text for t in tokens)) == 1:
        return max(tokens, key=lambda t: t.probability)

    similarities = []
    # print(f"Selecting best token from group of {len(tokens)} tokens")
    for t in tokens:
        # print(f"\tToken: {t.text}")

        prev_sim = (
            1
            if prev is None
            else torch.nn.functional.cosine_similarity(t.embedding, prev, dim=0).item()
        )
        prev_sim = (prev_sim + 1) / 2  # Normalize to [0, 1]
        sim = (prev_sim * p) * (t.probability * c)

        # print(
        #     f"\t\tMean similarity: {mean_sim}, Previous similarity: {prev_sim},  Combined: {sim}"
        # )
        similarities.append(sim)

    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    return tokens[best_idx]


def select_best_confidence_and_prev_and_mean(
    group: list[Token],
    prev: torch.Tensor | None,
    m: float = 1,
    p: float = 1,
    c: float = 1,
) -> Token:
    tokens = [t for t in group if t.is_word]
    if not tokens:
        return group[0]
    elif len(set(t.text for t in tokens)) == 1:
        return max(tokens, key=lambda t: t.probability)

    tensors = [t.embedding for t in tokens]
    mean = torch.mean(torch.stack(tensors), dim=0)

    similarities = []
    # print(f"Selecting best token from group of {len(tokens)} tokens")
    for t in tokens:
        # print(f"\tToken: {t.text}")

        mean_sim = torch.nn.functional.cosine_similarity(
            mean, t.embedding, dim=0
        ).item()
        mean_sim = (mean_sim + 1) / 2  # Normalize to [0, 1]

        prev_sim = (
            1
            if prev is None
            else torch.nn.functional.cosine_similarity(t.embedding, prev, dim=0).item()
        )
        prev_sim = (prev_sim + 1) / 2  # Normalize to [0, 1]

        sim = (mean_sim * m) * (prev_sim * p) * (t.probability * c)

        # print(
        #     f"\t\tMean similarity: {mean_sim}, Previous similarity: {prev_sim},  Combined: {sim}"
        # )
        similarities.append(sim)

    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    return tokens[best_idx]


def group_similar_tokens(
    source: list[Token],
    token_groups: list[list[Token]],
    padding: int,
    iou_threshold: float,
    cos_threshold: float,
    smooth: float,
) -> tuple[list[list[Token]], list[Token]]:
    orphan_tokens = []
    group_idx = 0
    new_token_groups = [[t for t in tg] for tg in token_groups]
    # print(f"similarity grouping")
    for t in source:
        if not t.is_word:
            orphan_tokens.append(t)
            continue
        # print(f"\tProcessing token: {t.text}")

        ss = {}
        for i in range(group_idx, len(token_groups)):
            for gt in token_groups[i]:
                iou = __token_iou(t, gt, padding, smooth)
                # print(f"\t\tToken: {t.text}, Group Token: {gt.text}: IOU: {iou}")
                if iou < iou_threshold:
                    if gt.start < t.start or gt.end < t.end:
                        continue
                    else:
                        break
                else:
                    s = __cosine_similarity(t, gt)
                    ss[i] = ss.get(i, [])
                    ss[i].append(s)
                    # print(f"\t\t\tSimilarity: {s}")

        if not ss:
            orphan_tokens.append(t)
            continue

        # print(f"\t\tSimilarities: {ss}")
        ss = {i: reduce(mul, s, 1) for i, s in ss.items()}
        max_arg = max(ss.keys(), key=lambda i: ss[i])
        if ss[max_arg] < cos_threshold:
            orphan_tokens.append(t)
            continue
        new_token_groups[max_arg].append(t)
        group_idx = max_arg

    return new_token_groups, orphan_tokens


def new_group_tokens(token_groups: list[list[Token]], orphan_tokens: list[Token]):
    orphan_token_groups = [[ot] for ot in orphan_tokens]
    token_groups.extend(orphan_token_groups)
    token_groups.sort(key=lambda tg: reduce(add, [t.start for t in tg], 0) / len(tg))
    return token_groups


def select_tokens(
    token_groups: list[list[Token]],
    select_func: tuple[
        Callable[[list[Token], torch.Tensor | None], Token]
    ] = select_best_only_prev,
) -> list[Token]:
    tokens = []
    prev_embedding = None
    prev_token: Token = None
    for tg in token_groups:
        best = select_func(tg, prev_embedding)
        if best.is_word:
            prev_embedding = (
                best.embedding
                if prev_embedding is None
                else torch.mean(torch.stack([prev_embedding, best.embedding]), dim=0)
            )
            if prev_token and best.start < prev_token.start:
                best.start = prev_token.end
        tokens.append(best)

    return tokens


def filter_token_groups(token_groups: list[list[Token]], time: int, n: int):
    token_groups = [[t for t in tg if t.is_word] for tg in token_groups]
    token_groups = [tg[-n:] for tg in token_groups if tg]
    return [tg for tg in token_groups if all(t.end > time for t in tg)]


__all__ = [
    "group_similar_tokens",
    "select_tokens",
    "new_group_tokens",
    "select_best_only_confidence",
    "select_best_only_prev",
    "select_best_confidence_and_prev",
    "select_best_confidence_and_prev_and_mean",
]
