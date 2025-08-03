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

from sj_utils.file.json import JsonSaver, load_json
from sj_utils.file.yaml import load_yaml
from sj_utils.collection import SafetyDict

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

SOURCE = "/workspaces/dev/datasets/ESIC-v1.1/v1.1/dev"
STORAGE = "/workspaces/dev/storage/esic/"
HYPERPARAMETER = "/workspaces/dev/test/performance_test/esic/hyperparameters/20250731/step1_16b-96k-3090"
OUTPUT_PATH = (
    "/workspaces/dev/test/performance_test/esic/output/20250731/step1_16b-96k-3090"
)
ESIC_VAL = "/workspaces/dev/test/performance_test/esic/val.json"

DESCRIPTION = """
20250731/step1_16b-96k-3090 테스트
RTX4070 테스트
새로운 노멀라이저 추가
세이브로더 사용
"""

src = Path(SOURCE)
storage = Path(STORAGE)
hyperparameter_path = Path(HYPERPARAMETER)
output_path = Path(OUTPUT_PATH)

data_paths = [Path(p) for p in load_json(Path(ESIC_VAL))[1]]
json_saver = JsonSaver(DESCRIPTION)


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
