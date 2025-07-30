import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/modules/python-utils",
    "/workspaces/dev/modules/ai-utils",
    "/workspaces/dev/test/modules/whisper_streaming",
    "/workspaces/dev",
    "/workspaces/dev/test/performance_test/esic",
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
)
from sj_utils.file.yaml import load_yaml
from sj_utils.file.json import JsonSaver
from sj_utils.collection import SafetyDict
from sj_ai_utils.datasets.esic_v1 import search_all_data


MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000
RANDOM_SEED = 42

MAX_COUNT = 1
TEST_ALL = True
USE_SAVE_LOADER = True

DESCRIPTION = """
테스트
"""

SOURCE = "/workspaces/dev/datasets/ESIC-v1.1/v1.1/test"
STORAGE = "/workspaces/dev/storage/esic/"
HYPERPARAMETER = "/workspaces/dev/test/performance_test/esic/hyperparameters/20250727/96000/trial_wer4o6_2010_20250727_024423.yaml"
OUTPUT_PATH = "/workspaces/dev/test/performance_test/esic/output/20250727/test.json"

# result_key = ["rt_whisper"]
result_key = ["whisper", "rt_whisper", "whisper_streaming"]

src = Path(SOURCE)
storage = Path(STORAGE)
json_saver = JsonSaver(DESCRIPTION)

data_paths = search_all_data(src)

def whisper_streaming():
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    print("Running Whisper Streaming...")

    asr = FasterWhisperASR("en", MODEL_SIZE)
    asr.use_vad()
    online = OnlineASRProcessor(asr)
    rng = np.random.default_rng(RANDOM_SEED)

    transcriber = get_whisper_streaming_transcriber(online, SAMPLE_RATE, rng)

    result = (
        test_process_all(data_paths, transcriber, max_count=MAX_COUNT)
        if TEST_ALL
        else test_process_each(data_paths, transcriber, max_count=MAX_COUNT)
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
    overlap = hyperparameter["asr"]["max_overlap_duration"]
    rng = np.random.default_rng(RANDOM_SEED)

    if USE_SAVE_LOADER:
        transcriber = get_token_saver_loader_transcriber(
            src, storage, SAMPLE_RATE, rng, hyperparameter, overlap
        )
    else:
        from rt_whisper import streamers

        token_streamer = streamers.get_token_streamer_with_vad_v2_min_filter(
            hyperparameter=hyperparameter,
        )
        transcriber = get_rt_whisper_transcriber(token_streamer, SAMPLE_RATE, rng)

    result = (
        test_process_all(data_paths, transcriber, max_count=MAX_COUNT)
        if TEST_ALL
        else test_process_each(data_paths, transcriber, max_count=MAX_COUNT)
    )

    del transcriber

    return result


def whisper():
    from faster_whisper import WhisperModel

    print("Running Whisper...")

    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    transcriber = get_faster_whisper_transcriber(model, SAMPLE_RATE)

    result = (
        test_process_all(data_paths, transcriber, max_count=MAX_COUNT)
        if TEST_ALL
        else test_process_each(data_paths, transcriber, max_count=MAX_COUNT)
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
