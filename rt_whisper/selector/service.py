from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from functools import lru_cache

if TYPE_CHECKING:
    from rt_whisper.data import Token


def group_similar_tokens(
    source: list[Token],
    token_groups: list[list[Token]],
    padding: int,
    iou_threshold: float,
    cos_threshold: float,
    smooth: float,
) -> tuple[list[list[Token]], list[Token]]:
    orphan_tokens = []
    idx_group = 0
    # print(f"similarity grouping")
    for token in source:
        if not token.is_word:
            orphan_tokens.append(token)
            continue

        # print(f"\tProcessing token: {token.text}")

        similarities = []
        for i in range(idx_group, len(token_groups)):
            group_token = token_groups[i][0]
            iou = __token_iou(token, group_token, padding, smooth)
            # print(f"\t\tToken: {token.text}, Group Token: {group_token.text}: IOU: {iou}")
            if iou < iou_threshold:
                if group_token.start < token.start or group_token.end < token.end:
                    continue
                else:
                    break
            else:
                if token.text == group_token.text:
                    similarity = 1.0
                else:
                    similarity = __cosine_similarity(token, group_token)
                similarities.append((i, similarity))
                # print(f"\t\t\tSimilarity: {similarity}")

        if not similarities:
            orphan_tokens.append(token)
            continue

        # print(f"\t\tSimilarities: {similarities}")
        max_arg = max(range(len(similarities)), key=lambda i: similarities[i][1])
        if similarities[max_arg][1] < cos_threshold:
            orphan_tokens.append(token)
            continue
        token_groups[similarities[max_arg][0]].append(token)
        idx_group = similarities[max_arg][0]

    return token_groups, orphan_tokens


def merge_tokens(
    token_groups: list[list[Token]],
    orphan_tokens: list[Token],
) -> list[Token]:
    best_tokens = []
    prev = None
    for tg in token_groups:
        best = __select_best(tg, prev)
        prev = (
            best.embedding
            if prev is None
            else torch.mean(torch.stack([prev, best.embedding]), dim=0)
        )
        best_tokens.append(best)
    tokens = best_tokens + orphan_tokens
    tokens.sort(key=lambda t: t.start if t.is_word else t.end)

    return tokens


def __select_best(group: list[Token], prev: torch.Tensor | None) -> Token:
    tokens = [t for t in group if t.is_word]
    if not tokens:
        return group[0]
    elif len(set(t.text for t in tokens)) == 1:
        return max(tokens, key=lambda t: t.probability)

    # NOTE 만약 이전 그룹만을 고려하지 않고, 중첩되는 모든 상황을 고려한다면 이 대표성이 중요할 수 있다. 따라서 주석만 함.
    # tensors = [t.embedding for t in tokens]
    # mean = torch.mean(torch.stack(tensors), dim=0)

    similarities = []
    # print(f"Selecting best token from group of {len(tokens)} tokens")
    for t in tokens:
        # print(f"\tToken: {t.text}")

        # mean_sim = torch.nn.functional.cosine_similarity(
        #     mean, t.embedding, dim=0
        # ).item()

        prev_sim = (
            1
            if prev is None
            else torch.nn.functional.cosine_similarity(t.embedding, prev, dim=0).item()
        )

        # mean_sim = (mean_sim + 1) / 2  # Normalize to [0, 1]

        prev_sim = (prev_sim + 1) / 2  # Normalize to [0, 1]

        # sim = mean_sim * t.probability * prev_sim

        sim = t.probability * prev_sim

        # print(
        # f"\t\tMean similarity: {mean_sim}, Previous similarity: {prev_sim}, Probability: {t.probability}, Combined: {sim}"
        # )
        similarities.append(sim)

    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    return tokens[best_idx]


def __cosine_similarity(A: Token, B: Token) -> float:
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


__all__ = ["group_similar_tokens", "merge_tokens"]
