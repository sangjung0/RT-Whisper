from __future__ import annotations
from typing import TYPE_CHECKING

from sj_utils.string import remove_spaces_and_symbols
from rt_whisper.data import Sentence

if TYPE_CHECKING:
    from rt_whisper import RTWhisperLogger
    from rt_whisper.data import Token
    from typing import Callable, Iterable


def select_language(language: str | None, tokens: list[Token]) -> str:
    return language or max((t.lang for t in tokens if t.is_word))


def cut_by_eos(tokens: list[Token]):
    segments = []

    start = 0
    for i, token in enumerate(tokens):
        if not token.is_word:
            segments.append(tokens[start:i])
            start = i + 1

    return segments


def cut_by_tokenizer(
    tokenizer: Callable[[str], Iterable[str]],
    tokens: list[Token],
    logger: RTWhisperLogger,
):
    segments = []

    word_tokens = [t for t in tokens if t.is_word]
    text = "".join(token.text for token in word_tokens)
    start_idx = 0
    for sent in tokenizer.segment(text):
        len_sent = len(remove_spaces_and_symbols(sent))
        if len_sent == 0:
            continue
        for i, token in enumerate(word_tokens[start_idx:]):
            len_sent -= len(remove_spaces_and_symbols(token.text))
            if len_sent <= 0:
                if len_sent < 0:
                    logger.warning(
                        f"Tokenizer did not match the segment length \n\t len_sent:{len_sent} \n\t sent: {sent} \n\t tokens: {[t.text for t in word_tokens[start_idx: i + start_idx + 1]]} \n\t normalize sent:'{remove_spaces_and_symbols(sent)}' \n\t normalize tokens:{[remove_spaces_and_symbols(t.text) for t in word_tokens[start_idx: i + start_idx + 1]]}",
                        group_level=3,
                    )
                segments.append(word_tokens[start_idx : i + start_idx + 1])
                start_idx = i + start_idx + 1
                break
        else:
            assert True, "Tokenizer did not match the segment length"
            segments.append(word_tokens[start_idx:])

    return segments


def tokens_to_sentences(
    segments: list[list[Token]], order: int
) -> tuple[list[Sentence], int]:
    sentences = []

    for tokens in segments:
        lang = list(set(token.lang for token in tokens))
        text = "".join(token.text for token in tokens)
        if len(text) == 0:
            continue
        sentences.append(
            Sentence(
                order=order,
                lang=lang,
                text=text,
                tokens=tokens,
            )
        )
        order += 1

    return sentences, order


def classify_candidate_completed(sentences: list[Sentence], anchor_timestamp: int):
    completed = [s for s in sentences if s.tokens[-1].end <= anchor_timestamp]
    candidate = [s for s in sentences if s.tokens[-1].end > anchor_timestamp]

    return completed, candidate


def context_tokens(candidate: list[Sentence], anchor_timestamp: int):
    completed_tokens = []

    for sentence in candidate:
        if sentence.tokens[0].start > anchor_timestamp:
            break
        for token in sentence.tokens:
            if token.end <= anchor_timestamp:
                completed_tokens.append(token)
            else:
                break

    return completed_tokens


__all__ = [
    "select_language",
    "cut_by_eos",
    "cut_by_tokenizer",
    "tokens_to_sentences",
    "classify_candidate_completed",
    "context_tokens",
]
