from __future__ import annotations
from typing import TYPE_CHECKING
from logging import Logger

from rt_whisper.core import logger_wrap
from rt_whisper.data import Sentence
from rt_whisper.abstracts import Worker
from rt_whisper.tokenizers import Tokenizer

from .data import ComposerParam, ComposerResult

if TYPE_CHECKING:
    from rt_whisper.data import TokenContext, Token


class Composer(Worker):

    @logger_wrap
    def __init__(self, logger: Logger):
        super().__init__()
        self.logger = logger

    # override
    def _can_process(self, context: TokenContext) -> ComposerParam:
        if (
            len(context.composer.prev.completed_tokens) > 0
            or len(context.segment_tokens) > 0
        ):
            return ComposerParam.from_context(context)
        return None

    # override
    def _process(self, param: ComposerParam) -> ComposerResult:
        tokens = param.prev_completed_tokens + param.segment_tokens
        completed_tokens = [t for t in tokens if t.end <= param.anchor_timestamp]
        segment_tokens = [t for t in tokens if t not in completed_tokens]

        completed_length = len(completed_tokens)
        source_tokens = completed_tokens + segment_tokens

        language = param.language or max((t.lang for t in source_tokens if t.is_word))
        tokenizer = Tokenizer.get_tokenizer(language)

        if tokenizer is None:
            completed, candidate, recycle_completed_tokens, order = self._cut_by_eos(
                source_tokens, param.order, completed_length
            )
        else:
            completed, candidate, recycle_completed_tokens, order = (
                self._cut_by_tokenizer(
                    tokenizer, source_tokens, param.order, completed_length
                )
            )

        recycle_segment_tokens = [t for t in segment_tokens if t.is_word]

        return ComposerResult(
            completed=completed,
            candidate=candidate,
            recycle_segment_tokens=recycle_segment_tokens,
            recycle_completed_tokens=recycle_completed_tokens,
            order=order,
        )

    # override
    def _update(self, context: TokenContext, result: ComposerResult):
        result.update_context(context)

    def _cut_by_eos(self, tokens: list[Token], order: int, completed_length: int):
        self.logger.warning("Tokenizer not found. Using default EOS.")

        completed = []
        candidate = []
        language = set()

        completed_index = 0
        start_idx = 0
        limit_completed = completed_length
        for idx in range(len(tokens)):
            limit_completed -= 1
            token = tokens[idx]
            language.add(token.lang)

            if not token.is_word:
                sentence = Sentence(
                    order=order,
                    lang=list(language),
                    text=token.text,
                    tokens=tokens[start_idx:idx],
                )
                if limit_completed < 0:
                    candidate.append(sentence)
                else:
                    completed.append(sentence)
                    completed_index = idx + 1

                language = set()
                start_idx = idx + 1
                order += 1

        return (
            completed,
            candidate,
            tokens[completed_index:completed_length],
            order - len(candidate),
        )

    def _cut_by_tokenizer(
        self, tokenizer, tokens: list[Token], order: int, completed_length: int
    ):

        completed_length -= len([t for t in tokens[:completed_length] if not t.is_word])
        tokens = [t for t in tokens if t.is_word]

        text = ""
        for token in tokens:
            text += token.text

        completed = []
        candidate = []
        completed_idx = 0
        start_idx = 0
        limit_completed = completed_length
        for sent in tokenizer.segment(text):
            len_scent = len(sent)
            for idx in range(start_idx, len(tokens)):
                len_scent -= len(tokens[idx].text)
                if len_scent <= 0:
                    end_idx = idx + 1
                    break
            else:
                end_idx = len(tokens)

            limit_completed = limit_completed - end_idx + start_idx
            sent_tokens = tokens[start_idx:end_idx]
            language = list(set([token.lang for token in sent_tokens]))
            sentence = Sentence(
                order=order, lang=language, text=sent, tokens=sent_tokens
            )
            if limit_completed < 0:
                candidate.append(sentence)
            else:
                completed.append(sentence)
                completed_idx = end_idx

            start_idx = end_idx
            order += 1

        return (
            completed,
            candidate,
            tokens[completed_idx:completed_length],
            order - len(candidate),
        )
