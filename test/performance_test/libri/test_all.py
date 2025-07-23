import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/modules/python-utils",
    "/workspaces/dev/modules/ai-utils",
    "/workspaces/dev/test/modules/whisper_streaming",
    "/workspaces/dev",
    "/workspaces/dev/test/performance_test/libri",
]
for path in paths:
    sys.path.append(os.path.abspath(path))
print(f"Current Python version: {sys.version}")

import numpy as np

from pathlib import Path

from util import (
    get_whisper_streaming_transcriber,
    get_rt_whisper_transcriber,
    get_token_saver_loader_transcriber,
    get_faster_whisper_transcriber,
    test_process_all,
    test_process_each,
    normalize_text,
)
from sj_utils.file.yaml import load_yaml
from sj_utils.file.json import JsonSaver
from sj_utils.collection import SafetyDict

MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000
RANDOM_SEED = 42

MAX_COUNT = 1
TEST_ALL = True
USE_SAVE_LOADER = True

DESCRIPTION = """
"""

SOURCE = "/workspaces/dev/datasets/LibriSpeechASRcorpus/test/test-clean/"
STORAGE = "/workspaces/dev/storage/libri/"
HYPERPARAMETER = "./hyperparameters/libri/sentence_error_47_4.yml"
OUTPUT_PATH = "/workspaces/dev/output/libri/clean/time_overall.json"

# result_key = ["rt_whisper"]
result_key = ["whisper", "rt_whisper", "whisper_streaming"]

src = Path(SOURCE)
storage = Path(STORAGE)
json_saver = JsonSaver(DESCRIPTION)


def whisper_streaming():
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    print("Running Whisper Streaming...")

    asr = FasterWhisperASR("en", MODEL_SIZE)
    asr.use_vad()
    online = OnlineASRProcessor(asr)
    rng = np.random.default_rng(RANDOM_SEED)

    transcriber = get_whisper_streaming_transcriber(online, SAMPLE_RATE, rng)

    result = (
        test_process_all(src, transcriber, normalize_text, MAX_COUNT)
        if TEST_ALL
        else test_process_each(src, transcriber, normalize_text, MAX_COUNT)
    )

    del transcriber
    del online
    del asr

    return result


def rt_whisper():
    print("Running RT Whisper...")

    hyperparameter = Path(HYPERPARAMETER)
    _, hyperparameter = load_yaml(hyperparameter)
    hyperparameter = SafetyDict(hyperparameter)

    if USE_SAVE_LOADER:
        _transcriber = get_token_saver_loader_transcriber(src, storage, SAMPLE_RATE)
        transcriber = lambda flac, transcribe_time: _transcriber(
            flac, storage, hyperparameter, transcribe_time
        )
    else:
        from rt_whisper import streamers

        token_streamer = streamers.get_token_streamer_with_vad_v2_min_filter(
            hyperparameter=hyperparameter,
        )
        rng = np.random.default_rng(RANDOM_SEED)
        transcriber = get_rt_whisper_transcriber(token_streamer, rng, SAMPLE_RATE)

    result = (
        test_process_all(src, transcriber, normalize_text, MAX_COUNT)
        if TEST_ALL
        else test_process_each(src, transcriber, normalize_text, MAX_COUNT)
    )

    del transcriber

    return result


def whisper():
    from faster_whisper import WhisperModel

    print("Running Whisper...")

    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    transcriber = get_faster_whisper_transcriber(model, SAMPLE_RATE)

    result = (
        test_process_all(src, transcriber, normalize_text, MAX_COUNT)
        if TEST_ALL
        else test_process_each(src, transcriber, normalize_text, MAX_COUNT)
    )

    del transcriber
    del model

    return result


if __name__ == "__main__":
    print("Starting performance tests...")

    results = {}
    for key in result_key:
        if key == "whisper":
            results[key] = whisper()
        elif key == "rt_whisper":
            results[key] = rt_whisper()
        elif key == "whisper_streaming":
            results[key] = whisper_streaming()

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_saver.save(results, output_path)

    print("Performance tests completed.")
