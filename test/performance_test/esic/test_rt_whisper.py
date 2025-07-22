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

from sj_utils.file.json import JsonSaver
from sj_utils.file.yaml import load_yaml
from sj_utils.collection import SafetyDict

from util import (
    get_token_saver_loader_transcriber,
    get_rt_whisper_transcriber,
    test_process_all,
    test_process_each,
    normalize_text,
)


MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000

MAX_COUNT = 2
TEST_ALL = True
USE_TOKEN_SAVER_LOADER = True

SOURCE = "/workspaces/dev/datasets/ESIC-v1.1/v1.1/test"
STORAGE = "/workspaces/dev/storage/esic/"
HYPERPARAMETER = "./hyperparameters/esic/20250722"
OUTPUT_PATH = "/workspaces/dev/output/esic/20250722"

DESCRIPTION = """
"""

src = Path(SOURCE)
storage = Path(STORAGE)
hyperparameter = Path(HYPERPARAMETER)
output_path = Path(OUTPUT_PATH)

json_saver = JsonSaver(DESCRIPTION)


def transcribe(hyperparameter_path: Path):
    print("Running RT Whisper...")

    _, hyperparameters = load_yaml(hyperparameter_path)
    hyperparameters = SafetyDict(hyperparameters)
    overlap = hyperparameters["asr"]["max_overlap_duration"]

    if USE_TOKEN_SAVER_LOADER:
        _transcriber = get_token_saver_loader_transcriber(src, storage, SAMPLE_RATE)
        transcriber = lambda mp4, transcribe_time: _transcriber(
            mp4, transcribe_time, hyperparameters, overlap
        )
    else:
        from rt_whisper import streamers

        token_streamer = streamers.get_token_streamer_with_vad_v2(hyperparameters)
        rng = np.random.default_rng(42)
        transcriber = get_rt_whisper_transcriber(token_streamer, rng, SAMPLE_RATE)

    result = (
        test_process_all(src, transcriber, normalize_text, MAX_COUNT)
        if TEST_ALL
        else test_process_each(src, transcriber, normalize_text, MAX_COUNT)
    )

    return result


if __name__ == "__main__":
    print("Starting performance tests...")

    if hyperparameter.exists() and hyperparameter.is_file():
        raise ValueError(
            f"Expected a directory for hyperparameter, but got a file: {hyperparameter}"
        )
    if output_path.exists() and output_path.is_file():
        raise ValueError(
            f"Expected a directory for output, but got a file: {output_path}"
        )

    hyper_paths = list(hyperparameter.glob("*.yaml"))

    for hyper_path in hyper_paths:
        print(f"Testing with hyperparameter: {hyper_path.name}")
        results = transcribe(hyper_path)
        output = output_path / f"{hyper_path.stem}.json"
        json_saver.save(results, output)

    print("Performance tests completed.")
