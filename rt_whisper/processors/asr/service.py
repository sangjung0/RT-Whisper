from __future__ import annotations
from types import coroutine
from typing import TYPE_CHECKING

from rt_whisper.data import Token

if TYPE_CHECKING:
    from typing import Callable, Iterable, Any
    import numpy as np


def generate_overlap_context(
    merged_chunk: np.ndarray,
    current_chunk: np.ndarray,
    prev_chunk: np.ndarray,
    offset: int,
    max_overlap_duration: int,
) -> tuple[np.ndarray, int, int]:
    """Generate context data for ASR processing

    Args:
        merged_chunk (np.ndarray): Merged chunk containing both previous and current chunk data.
        current_chunk (np.ndarray): Current chunk.
        prev_chunk (np.ndarray): Previous chunk.
        offset (int): Offset relative to the current chunk.
        max_overlap_duration (int): Maximum duration of overlap to consider.

    Returns:
        tuple[np.ndarray, int, int]: A tuple containing:
            - context_chunk (np.ndarray): The context chunk to be used for ASR processing.
            - context_offset (int): The offset for the context chunk.
            - anchor_timestamp (int): The timestamp for the anchor position in the context chunk.
    """
    anchor = max(0, merged_chunk.shape[0] - max_overlap_duration)
    context_chunk = merged_chunk[anchor:]
    context_offset = offset + current_chunk.shape[0]
    anchor_timestamp = anchor + offset - prev_chunk.shape[0]

    return context_chunk, context_offset, anchor_timestamp


def adjust_anchor_timestamp(
    anchor_timestamp: int, segment_tokens: list[Token], padding: int = 1600
) -> int:
    for token in segment_tokens:
        if anchor_timestamp < token.start:
            return max(0, token.start - padding)
    return anchor_timestamp


def segment_to_token_list(
    segments: Iterable,
    language: str,
    offset: int,
    sample_rate: int,
    within_eos: bool,
    tokenizer_encoder: Callable[[str], list[int]],
) -> list[Token]:
    """Convert segments to a list of tokens.

    Args:
        segments (Iterable): Segments to convert.
        language (str): Language of the segments.
        offset (int): Offset to apply to the start and end times of the tokens.
        sample_rate (int): Sample rate of the audio data, used to convert time to sample indices.
        within_eos (bool): If True, include the start and end of the segment as a single token.
        tokenizer_encoder (Callable[[str], list[int]]): Function to encode text into tokens.

    Returns:
        list[Token]: A list of Token objects representing the segments.
    """
    segment_tokens = []
    for segment in segments:
        tokens = [
            Token(
                start=int(w.start * sample_rate) + offset,
                end=int(w.end * sample_rate) + offset,
                text=w.word,
                lang=language,
                tokens=tokenizer_encoder(w.word),
                probability=w.probability,
            )
            for w in segment.words
        ]
        if within_eos:
            start = tokens[0].start
            end = tokens[-1].end
            tokens.append(
                Token(
                    start=start,
                    end=end,
                    text=segment.text,
                    lang="",
                    tokens=[],
                    probability=1,
                    is_word=False,
                )
            )
        segment_tokens.extend(tokens)

    return segment_tokens


def transcribe(
    chunk: np.ndarray,
    language: str | None,
    prompt: str | None,
    transcriber: Callable[[np.ndarray, str, str], tuple[Iterable, Any]],
) -> tuple[Iterable, str]:
    """Transcribe audio chunk into segments and update language.

    Args:
        chunk (np.ndarray): Audio chunk to transcribe.
        language (str | None): Language of the audio chunk, or None if unknown.
        prompt (str | None): Prompt to guide the transcription, or None if not applicable.
        transcriber (Callable[[np.ndarray, str, str], tuple[Iterable, Any]]): Function to transcribe the audio chunk.

    Returns:
        tuple[list[Token], str]: A tuple containing:
            - segments (list): List of transcribed segments.
            - language (str): Updated language after transcription.
    """
    if chunk.shape[0] == 0:
        return [], language
    segments, info = transcriber(chunk, language, prompt)
    language = info.language or language
    return segments, language


async def async_transcribe(
    chunk: np.ndarray,
    language: str | None,
    transcriber: coroutine[Callable[[np.ndarray, str, str], tuple[Iterable, Any]]],
) -> tuple[Iterable, str]:
    if chunk.shape[0] == 0:
        return [], None
    segments, info = await transcriber(chunk)
    language = info.language or language
    return segments, language
