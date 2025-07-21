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
    normalize_text,
)


MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000
OVERLAP = 96000
MAX_COUNT = 2
TEST_ALL = True

SOURCE = "/workspaces/dev/datasets/ESIC-v1.1/v1.1/test"
STORAGE = "/workspaces/dev/storage/esic/"
HYPERPARAMETER = "./hyperparameters/esic/20250721/20250721_wer10o1v4.yml"
OUTPUT_PATH = "/workspaces/dev/output/esic/test.json"

# result_key = ["rt_whisper"]
result_key = ["whisper", "rt_whisper", "whisper_streaming"]

src = Path(SOURCE)
storage = Path(STORAGE)


def whisper_streaming():
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    print("Running Whisper Streaming...")

    asr = FasterWhisperASR("en", MODEL_SIZE)
    asr.use_vad()
    online = OnlineASRProcessor(asr)
    rng = np.random.default_rng(42)

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

    # from rt_whisper import streamers
    # token_streamer = streamers.get_token_streamer_with_vad_v2(
    #     hyperparameter=HYPERPARAMETER,
    # )
    # rng = np.random.default_rng(42)
    # transcriber = get_rt_whisper_transcriber(token_streamer, rng, SAMPLE_RATE)

    _transcriber = get_token_saver_loader_transcriber(src, storage, SAMPLE_RATE)
    transcriber = lambda mp4, transcribe_time: _transcriber(
        mp4, transcribe_time, HYPERPARAMETER, OVERLAP
    )

    result = (
        test_process_all(src, transcriber, normalize_text, MAX_COUNT)
        if TEST_ALL
        else test_process_each(src, transcriber, normalize_text, MAX_COUNT)
    )

    del transcriber
    del _transcriber

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
    import json

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
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Performance tests completed.")
