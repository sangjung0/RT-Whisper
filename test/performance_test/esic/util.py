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

import numpy as np

from functools import lru_cache
from pathlib import Path
from typing import Callable

from sj_utils.audio import load_audio_from_mp4
from sj_utils.evaluator import TimeChecker
from sj_ai_utils.datasets.esic_v1 import search_all_ref_and_hyp
from sj_utils.collection import SafetyDict

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
def load_mp4(mp4, sr):
    return load_audio_from_mp4(mp4, sr=sr)


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
    return gwst(online, sr, load_mp4, rng)


def get_rt_whisper_transcriber(
    token_streamer,
    sr: int,
    rng: np.random.Generator | np.random.RandomState = np.random,
) -> Callable[[Path, TimeChecker], str]:
    return grwt(token_streamer, sr, load_mp4, rng)


def get_faster_whisper_transcriber(
    model,
    sr: int,
) -> Callable[[Path, TimeChecker], str]:
    return gfwt(model, sr, load_mp4)


def get_token_saver_loader_transcriber(
    source: Path,
    storage: Path,
    sr: int,
    rng: np.random.Generator | np.random.RandomState = np.random,
    hyperparameter: SafetyDict = None,
    overlap: int = None,
) -> Callable[[Path, TimeChecker, SafetyDict, int], str]:
    return gtslt(
        source,
        storage,
        sr,
        load_mp4,
        rng,
        hyperparameter=hyperparameter,
        overlap=overlap,
    )


__all__ = [
    "test_process_all",
    "test_process_each",
    "load_mp4",
    "normalize_text",
    "get_whisper_streaming_transcriber",
    "get_rt_whisper_transcriber",
    "get_faster_whisper_transcriber",
    "get_token_saver_loader_transcriber",
]
