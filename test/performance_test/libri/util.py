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

from sj_ai_utils.datasets.libri_speech_asr_corpus import search_all_ref_and_hyp
from sj_utils.collection import SafetyDict
from sj_utils.evaluator import TimeChecker
from sj_utils.string import normalize_text_only_en

from common_util import (
    test_process_all as tpa,
    test_process_each as tpe,
    normalize_text as nt,
    get_whisper_streaming_transcriber as gwst,
    get_rt_whisper_transcriber as grwt,
    get_faster_whisper_transcriber as gfwt,
    get_token_saver_loader_transcriber as gtslt,
)


@lru_cache(maxsize=4196)
def load_audio(audio, sr):
    return librosa.load(audio, sr=sr)


def normalize_text(text: str) -> Path:
    return normalize_text_only_en(nt(text)).upper()


def test_process_all(
    src: Path,
    transcriber: Callable[[Path, TimeChecker], str],
    normalizer: Callable[[Path], Path] = normalize_text,
    max_count: int = -1,
) -> dict:
    return tpa(src, transcriber, search_all_ref_and_hyp, normalizer, max_count)


def test_process_each(
    src: Path,
    transcriber: Callable[[Path, TimeChecker], str],
    normalizer: Callable[[Path], Path] = normalize_text,
    max_count: int = -1,
) -> dict:
    return tpe(src, transcriber, search_all_ref_and_hyp, normalizer, max_count)


def get_whisper_streaming_transcriber(
    online,
    sr: int,
    rng: np.random.Generator | np.random.RandomState = np.random,
) -> Callable[[Path, TimeChecker], str]:
    return gwst(online, sr, load_audio, rng)


def get_rt_whisper_transcriber(
    token_streamer,
    sr: int,
    rng: np.random.Generator | np.random.RandomState = np.random,
) -> Callable[[Path, TimeChecker], str]:
    return grwt(token_streamer, sr, load_audio, rng)


def get_faster_whisper_transcriber(
    model,
    sr: int,
) -> Callable[[Path, TimeChecker], str]:
    return gfwt(model, sr, load_audio)


def get_token_saver_loader_transcriber(
    source: Path,
    storage: Path,
    sr: int,
    rng: np.random.Generator | np.random.RandomState = np.random,
    hyperparameter: SafetyDict = None,
    overlap: int = None,
) -> Callable[[Path, TimeChecker], str]:
    return gtslt(
        source,
        storage,
        sr,
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
