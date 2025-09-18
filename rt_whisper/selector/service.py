from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from functools import lru_cache, reduce
from operator import add
from rapidfuzz.distance import Levenshtein

if TYPE_CHECKING:
    from rt_whisper.data import Token


def __group_range_edit_distance(
    group: list[list[Token]],
    tokens: list[Token],
    g_idx: int,
    t_idx: int,
    window: int,
    alpha: float,
) -> float:
    assert window >= 0, "Range must be greater than 0"
    assert alpha > 0 and alpha <= 1, "Alpha must be between 0 and 1"

    weight = 1.0
    total_ed = __group_edit_distance(group[g_idx], tokens[t_idx]) * weight
    total_weight = weight
    for r in range(1, window + 1):
        weight *= alpha
        total_weight += weight
        lg, lt, rg, rt = g_idx - r, t_idx - r, g_idx + r, t_idx + r
        if lg >= 0 and lt >= 0:
            total_ed += __group_edit_distance(group[lg], tokens[lt]) * weight
        if rg < len(group) and rt < len(tokens):
            total_ed += __group_edit_distance(group[rg], tokens[rt]) * weight
    return total_ed / total_weight


def __group_edit_distance(group: list[Token], T: Token) -> float:
    if not group:
        return 0.0
    return sum(__edit_distance(t.text, T.text) for t in group) / len(group)


def __edit_distance(A: str, B: str) -> float:
    if A == B:
        return 0
    if A < B:
        A, B = B, A
    return __lru_edit_distance(A, B)


@lru_cache(maxsize=4096)
def __lru_edit_distance(A: str, B: str) -> float:
    return Levenshtein.normalized_distance(A, B)


def __group_cosine_similarity(group: list[Token], T: Token, eps: float) -> float:
    if not group:
        return 0.0
    return sum(__cosine_similarity(t, T, eps) for t in group) / len(group)


def __cosine_similarity(A: Token, B: Token, eps: float) -> float:
    if A.text == B.text:
        return 1
    if A.text < B.text:
        A, B = B, A
    return __lru_cosine_similarity(A, B, eps)


@lru_cache(maxsize=4096)
def __lru_cosine_similarity(A: Token, B: Token, eps: float) -> float:
    return __cosine_similarity_numpy(A.embedding, B.embedding, eps)


def __cosine_similarity_numpy(A: np.ndarray, B: np.ndarray, eps: float) -> float:
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + eps)


def __iou(s1: float, e1: float, s2: float, e2: float, smooth: float) -> float:
    inner = max(0, min(e1, e2) - max(s1, s2))
    outer = max(e1, e2) - min(s1, s2)

    return inner / (outer + smooth)


def __select_best(
    group: list[Token],
    prev: np.ndarray | None,
    m: float,
    p: float,
    c: float,
    g_cos_eps: float,
    cos_eps: float,
) -> Token:
    assert (
        0 <= m <= 1 and 0 <= p <= 1 and 0 <= c <= 1
    ), "m, p, c must be between 0 and 1"
    assert 1 - 1e-3 <= m + p + c <= 1, "m + p + c must be equal to 1"

    tokens = [t for t in group if t.is_word]
    if not tokens:
        return group[0]
    elif len(set(t.text for t in tokens)) == 1:
        return max(tokens, key=lambda t: t.probability)

    similarities = []
    # print(f"Selecting best token from group of {len(tokens)} tokens")
    for t in tokens:
        # print(f"\tToken: {t.text}")

        mean_sim = __group_cosine_similarity(tokens, t, g_cos_eps)
        mean_sim = (mean_sim + 1) / 2  # Normalize to [0, 1]

        prev_sim = (
            1 if prev is None else __cosine_similarity_numpy(prev, t.embedding, cos_eps)
        )
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
    s: float,
    i: float,
    ged_window: int,
    ged_alpha: float,
    iou_smooth: float = 1e-6,
) -> tuple[list[list[Token]], list[Token]]:
    assert padding >= 0, "padding must be non-negative"
    assert 0 <= iou_threshold <= 1, "iou_threshold must be between 0 and 1"
    assert 0 <= cos_threshold <= 1, "cos_threshold must be between 0 and 1"
    assert 0 <= s <= 1 and 0 <= i <= 1, "s and i must be between 0 and 1"
    assert s + i == 1, "s + i must be equal to 1"

    orphan_tokens = []
    group_idx = 0
    new_token_groups = [[t for t in tg] for tg in token_groups]
    # print(f"similarity grouping")
    for t_idx, t in enumerate(source):
        if not t.is_word:
            orphan_tokens.append(t)
            continue
        # print(f"\tProcessing token: {t.text}")

        ss = {}
        for g_idx in range(group_idx, len(token_groups)):
            group = token_groups[g_idx]
            group_start = sum(t.start for t in group) / len(group)
            group_end = sum(t.end for t in group) / len(group)

            gs = max(0, group_start - padding)
            ge = group_end + padding
            ts = max(0, t.start - padding)
            te = t.end + padding
            iou = __iou(gs, ge, ts, te, iou_smooth)
            # print(f"\t\tToken: {t.text}, Group Token: {gt.text}: IOU: {iou}")
            if iou < iou_threshold and group_start > t.start and group_end > t.end:
                break
            else:
                ed = 1 - __group_range_edit_distance(
                    token_groups, source, g_idx, t_idx, ged_window, ged_alpha
                )
                ss[g_idx] = ed * s + iou * i
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
    alpha: float,
    sb_m: float,
    sb_p: float,
    sb_c: float,
    g_cos_eps: float = 1e-6,
    cos_eps: float = 1e-6,
) -> list[Token]:
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"

    tokens = []
    prev_embedding = None
    prev_token: Token = None
    for tg in token_groups:
        best = __select_best(tg, prev_embedding, sb_m, sb_p, sb_c, g_cos_eps, cos_eps)
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


def completed_and_candidate_tokens(
    tokens: list[Token], anchor_timestamp: int
) -> tuple[list[Token], list[Token]]:
    assert anchor_timestamp >= 0, "Anchor timestamp must be non-negative"

    completed = [t for t in tokens if t.end < anchor_timestamp]
    candidate = tokens[len(completed) :]
    return completed, candidate


def filter_token_groups(token_groups: list[list[Token]], n: int, size: int):
    assert n >= 0, "n must be non-negative"
    assert size > 0, "size must be greater than 0"

    new = [[t for t in tg if t.is_word][-size:] for tg in token_groups[n:]]
    new = [tg for tg in new if tg]
    return new


__all__ = [
    "group_similar_tokens",
    "select_tokens",
    "new_group_tokens",
    "completed_and_candidate_tokens",
    "filter_token_groups",
]
