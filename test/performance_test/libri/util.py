import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/modules/python-utils",
    "/workspaces/dev/modules/ai-utils",
    "/workspaces/dev",
    "/workspaces/dev/test/modules/whisper_streaming",
    "/workspaces/dev/test/performance_test",
]
for path in paths:
    sys.path.append(os.path.abspath(path))

import librosa
import numpy as np

from pathlib import Path
from typing import Callable
from functools import lru_cache

from sj_ai_utils.datasets.libri_speech_asr_corpus.sclite import generate_trn
from sj_utils.collection import SafetyDict
from sj_utils.evaluator import TimeChecker

from common_util import (
    test_process_all as tpa,
    test_process_each as tpe,
    normalize_text,
    get_whisper_streaming_transcriber as gwst,
    get_rt_whisper_transcriber as grwt,
    get_faster_whisper_transcriber as gfwt,
    get_token_saver_loader_transcriber as gtslt,
)


@lru_cache(maxsize=4196)
def load_audio(audio, sr):
    return librosa.load(audio, sr=sr)


def test_process_all(
    data_paths: list[Path],
    transcriber: Callable[[Path, TimeChecker], str],
    normalizer: Callable[[Path], Path] = normalize_text,
    max_count: int = -1,
) -> dict:
    return tpa(data_paths, transcriber, generate_trn, normalizer, max_count)


def test_process_each(
    data_paths: list[Path],
    transcriber: Callable[[Path, TimeChecker], str],
    normalizer: Callable[[Path], Path] = normalize_text,
    max_count: int = -1,
) -> dict:
    return tpe(data_paths, transcriber, generate_trn, normalizer, max_count)


def get_whisper_streaming_transcriber(
    rng: np.random.Generator | np.random.RandomState = np.random,
    audio_chunk_mean: int = 48000,
    audio_chunk_std: int = 400,
    audio_chunk_min_max: float = 0.1,
) -> Callable[[Path, TimeChecker], str]:
    return gwst(
        load_audio,
        rng,
        audio_chunk_mean=audio_chunk_mean,
        audio_chunk_std=audio_chunk_std,
        audio_chunk_min_max=audio_chunk_min_max,
    )


def get_rt_whisper_transcriber(
    hyperparameter: SafetyDict,
    rng: np.random.Generator | np.random.RandomState = np.random,
    audio_chunk_mean: int = 48000,
    audio_chunk_std: int = 400,
    audio_chunk_min_max: float = 0.1,
) -> Callable[[Path, TimeChecker], str]:
    return grwt(
        hyperparameter,
        load_audio,
        rng,
        audio_chunk_mean=audio_chunk_mean,
        audio_chunk_std=audio_chunk_std,
        audio_chunk_min_max=audio_chunk_min_max,
    )


def get_faster_whisper_transcriber() -> Callable[[Path, TimeChecker], str]:
    return gfwt(load_audio)


def get_token_saver_loader_transcriber(
    source: Path,
    storage: Path,
    rng: np.random.Generator | np.random.RandomState = np.random,
    hyperparameter: SafetyDict = None,
    overlap: int = None,
) -> Callable[[Path, TimeChecker], str]:
    return gtslt(
        source,
        storage,
        load_audio,
        rng,
        hyperparameter=hyperparameter,
        overlap=overlap,
    )


__all__ = [
    "test_process_all",
    "test_process_each",
    "load_audio",
    "normalize_text",
    "get_whisper_streaming_transcriber",
    "get_rt_whisper_transcriber",
    "get_faster_whisper_transcriber",
    "get_token_saver_loader_transcriber",
]
