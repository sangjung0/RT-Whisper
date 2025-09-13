from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from functools import lru_cache, reduce
from operator import add
from rapidfuzz.distance import Levenshtein

if TYPE_CHECKING:
    from rt_whisper.data import Token


def __group_edit_distance(group: list[Token], T: Token) -> float:
    if not group:
        return 0.0
    return sum(Levenshtein.normalized_distance(t.text, T.text) for t in group) / len(
        group
    )


def __group_cosine_similarity(group: list[Token], T: Token) -> float:
    if not group:
        return 0.0
    return sum(__cosine_similarity(t, T) for t in group) / len(group)


def __cosine_similarity(A: Token, B: Token) -> float:
    if A.text == B.text:
        return 1
    if hash(A) < hash(B):
        A, B = B, A
    return __lru_cosine_similarity(A, B)


@lru_cache(maxsize=4096)
def __lru_cosine_similarity(A: Token, B: Token) -> float:
    return __cosine_similarity_numpy(A.embedding, B.embedding)


def __cosine_similarity_numpy(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> float:
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + eps)


def __iou(s1: float, e1: float, s2: float, e2: float, smooth: float = 1e-6) -> float:
    inner = max(0, min(e1, e2) - max(s1, s2))
    outer = max(e1, e2) - min(s1, s2)

    return inner / (outer + smooth)


def __select_best(
    group: list[Token],
    prev: np.ndarray | None,
    m: float = 1,
    p: float = 1,
    c: float = 1,
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

        mean_sim = __group_cosine_similarity(tokens, t)
        mean_sim = (mean_sim + 1) / 2  # Normalize to [0, 1]

        prev_sim = 1 if prev is None else __cosine_similarity_numpy(prev, t.embedding)
        prev_sim = (prev_sim + 1) / 2  # Normalize to [0, 1]

        sim = (mean_sim * m) + (prev_sim * p) + (t.probability * c)

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
    s: float = 1,
    i: float = 1,
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
        for idx in range(group_idx, len(token_groups)):
            group = token_groups[idx]
            group_start = max(t.start for t in group)
            group_end = min(t.end for t in group)

            gs = max(0, group_start - padding)
            ge = group_end + padding
            ts = max(0, t.start - padding)
            te = t.end + padding
            iou = __iou(gs, ge, ts, te, smooth)
            # print(f"\t\tToken: {t.text}, Group Token: {gt.text}: IOU: {iou}")
            if iou < iou_threshold and group_start > t.start and group_end > t.end:
                break
            else:
                ed = 1 - __group_edit_distance(group, t)
                ss[idx] = ed * s + iou * i
                # ss[i] = __group_cosine_similarity(group, t)
                # print(f"\t\t\tSimilarity: {s}")

        if not ss:
            orphan_tokens.append(t)
            continue

        # print(f"\t\tSimilarities: {ss}")
        max_arg = max(ss.keys(), key=lambda idx: ss[idx])
        # 더이상 cos_threshold 변수 명이 맞지 않지만, 일단 임시 사용
        if ss[max_arg] < cos_threshold:
            orphan_tokens.append(t)
            continue
        new_token_groups[max_arg].append(t)
        group_idx = max_arg

    return new_token_groups, orphan_tokens


def new_group_tokens(token_groups: list[list[Token]], orphan_tokens: list[Token]):
    orphan_token_groups = [[ot] for ot in orphan_tokens]
    token_groups.extend(orphan_token_groups)
    token_groups.sort(
        key=lambda tg: reduce(add, [t.start if t.is_word else t.end for t in tg], 0)
        / len(tg)
    )
    return token_groups


def select_tokens(
    token_groups: list[list[Token]],
    m: float = 1,
    p: float = 1,
    c: float = 1,
    alpha: float = 0.3,
) -> list[Token]:
    tokens = []
    prev_embedding = None
    prev_token: Token = None
    for tg in token_groups:
        best = __select_best(tg, prev_embedding, m, p, c)
        if best.is_word:
            prev_embedding = (
                best.embedding
                if prev_embedding is None
                else (1 - alpha) * prev_embedding + alpha * best.embedding
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
]
