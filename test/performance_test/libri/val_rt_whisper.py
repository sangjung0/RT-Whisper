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

from sj_utils.file.json import JsonSaver
from sj_utils.file.yaml import load_yaml
from sj_utils.collection import SafetyDict
from sj_ai_utils.datasets.libri_speech_asr_corpus.service import search_dirs

from util import (
    get_token_saver_loader_transcriber,
    get_rt_whisper_transcriber,
    test_process_all,
    test_process_each,
)


SEED = 42

MAX_COUNT = -1
TEST_ALL = True
USE_TOKEN_SAVER_LOADER = True

SOURCE = "/workspaces/dev/datasets/LibriSpeechASRcorpus/dev/dev-clean"
STORAGE = "/workspaces/dev/storage/libri/"
HYPERPARAMETER = (
    "/workspaces/dev/test/performance_test/esic/hyperparameters/20250731/step1_16b-96k"
)
OUTPUT_PATH = (
    "/workspaces/dev/test/performance_test/libri/output/dev/20250731/step1_16b-96k"
)

DESCRIPTION = """
20250731/step1_16b-96k 검증
RTX4070 테스트
세이브로더사용
"""

src = Path(SOURCE)
storage = Path(STORAGE)
hyperparameter_path = Path(HYPERPARAMETER)
output_path = Path(OUTPUT_PATH)

json_saver = JsonSaver(DESCRIPTION)
data_paths = search_dirs(src)


def transcribe(hyperparameter_path: Path):
    print("Running RT Whisper...")

    _, hyperparameter = load_yaml(hyperparameter_path)
    hyperparameter = SafetyDict(hyperparameter)
    overlap = hyperparameter["asr"]["max_overlap_duration"]
    rng = np.random.default_rng(SEED)

    if USE_TOKEN_SAVER_LOADER:
        transcriber = get_token_saver_loader_transcriber(
            src, storage, rng, hyperparameter, overlap
        )
    else:
        transcriber = get_rt_whisper_transcriber(hyperparameter, rng)

    result = (
        test_process_all(data_paths, transcriber, max_count=MAX_COUNT)
        if TEST_ALL
        else test_process_each(data_paths, transcriber, max_count=MAX_COUNT)
    )

    return result


if __name__ == "__main__":
    print("Starting performance tests...")

    if hyperparameter_path.exists() and hyperparameter_path.is_file():
        raise ValueError(
            f"Expected a directory for hyperparameter, but got a file: {hyperparameter_path}"
        )
    if output_path.exists() and output_path.is_file():
        raise ValueError(
            f"Expected a directory for output, but got a file: {output_path}"
        )

    hyper_paths = list(hyperparameter_path.glob("*.yaml"))

    for hyper_path in hyper_paths:
        print(f"Testing with hyperparameter: {hyper_path.name}")
        results = transcribe(hyper_path)
        output = output_path / f"{hyper_path.stem}.json"
        json_saver.save(results, output)

    print("Performance tests completed.")
