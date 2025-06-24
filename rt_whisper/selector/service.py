from __future__ import annotations
from typing import TYPE_CHECKING
from rapidfuzz.distance import Levenshtein

if TYPE_CHECKING:
    from rt_whisper.data import Token


def position_tokens(
    source: list[Token],
    target: list[Token],
    tolerance: int = 0,
):
    token_groups = []
    rest = []
    start = source[-1].start + tolerance
    for t in target:
        if (not t.is_word and t.end > start) or (t.is_word and t.start > start):
            rest.append(t)
        else:
            token_groups.append([t])

    return token_groups, rest


def group_similar_tokens(
    source: list[Token],
    token_groups: list[list[Token]],
    search_range: int,
    padding: int,
    threshold: float,
    smooth: float,
):
    orphan_tokens = []
    idx, idx_group = -1, 0
    while True:
        idx += 1
        if idx >= len(source) or idx_group >= len(token_groups):
            break

        token = source[idx]
        similarities = []
        for i in range(idx_group, len(token_groups)):
            group_token = token_groups[i][0]
            if group_token.start < token.start - search_range:
                continue
            elif group_token.start > token.start + search_range:
                break
            elif not token.is_word:
                similarities.append((i, 0))
            else:
                similarity = __token_similarity(token, group_token, padding, smooth)
                similarities.append((i, similarity))

        if not similarities:
            orphan_tokens.append(token)
            continue

        max_arg = max(range(len(similarities)), key=lambda i: similarities[i][1])
        if similarities[max_arg][1] < threshold:
            orphan_tokens.append(token)
            continue
        token_groups[similarities[max_arg][0]].append(token)
        idx_group = similarities[max_arg][0]

    for i in range(idx, len(source)):
        orphan_tokens.append(source[i])

    return token_groups, orphan_tokens


def merge_tokens(
    token_groups: list[list[Token]],
    orphan_tokens: list[Token],
    rest: list[Token],
):
    tokens = [max(tk, key=lambda x: x.probability) for tk in token_groups]
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

    return tokens + rest


def __token_similarity(A: Token, B: Token, padding: int, smooth: float) -> float:
    # ratio = fuzz.ratio(A.text.strip().lower(), B.text.strip().lower())
    ratio = Levenshtein.normalized_similarity(A.tokens, B.tokens)
    iou = __token_iou(A, B, padding, smooth)
    return (ratio + iou) / 2


def __token_iou(A: Token, B: Token, padding: int = 3200, smooth: float = 1e-6) -> float:
    a1 = max(0, A.start - padding)
    b1 = A.end + padding
    a2 = max(0, B.start - padding)
    b2 = B.end + padding

    inner = max(0, min(b1, b2) - max(a1, a2))
    outer = max(max(b1, b2) - min(a1, a2), smooth)

    return inner / outer
