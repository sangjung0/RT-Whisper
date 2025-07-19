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
    tokens = [__select_best(tk) for tk in token_groups]
    t_idx = 0
    o_idx = 0
    for o_idx in range(len(orphan_tokens)):
        o_token = orphan_tokens[o_idx]
        while t_idx < len(tokens):
            token = tokens[t_idx]
            if o_token.is_word and token.start > o_token.start:
                tokens.insert(t_idx, o_token)
                t_idx += 2
                break
            elif not o_token.is_word and token.end > o_token.end:
                tokens.insert(t_idx, o_token)
                t_idx += 2
                break
            t_idx += 1
        break
    for o_idx in range(o_idx, len(orphan_tokens)):
        tokens.append(orphan_tokens[o_idx])

    return tokens


def __select_best(group: list[Token]) -> Token:
    tokens = [t for t in group if t.is_word]
    if not tokens:
        return group[0]
    elif len(set(t.text for t in tokens)) == 1:
        return max(tokens, key=lambda t: t.probability)

    tensors = [t.embedding for t in tokens]
    mean = torch.mean(torch.stack(tensors), dim=0)
    similarities = [
        torch.nn.functional.cosine_similarity(mean, t.embedding, dim=0).item()
        * t.probability
        for t in tokens
    ]
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
