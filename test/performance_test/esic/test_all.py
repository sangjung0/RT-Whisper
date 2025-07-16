import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "./modules/python-utils",
    "./modules/ai-utils",
    "./test/modules/whisper_streaming",
    "/workspaces/dev",
]
for path in paths:
    sys.path.append(os.path.abspath(path))
print(f"Current Python version: {sys.version}")

import numpy as np

from pathlib import Path
from typing import Callable
from functools import lru_cache

from sj_ai_utils.asr.whisper_utils import *
from sj_ai_utils.datasets.esic_v1 import *
from sj_ai_utils.evaluator.sclite_utils import *
from sj_utils.audio_utils import *
from sj_utils.string_utils import *
from sj_utils.evaluator import TimeChecker

MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000
SOURCE = "/workspaces/dev/datasets/ESIC-v1.1/v1.1/test"
HYPERPARAMETER = "./hyperparameters/libri/sentence_error_47_4.yml"
OUTPUT_PATH = "/workspaces/dev/output/esic/v_overall.json"

src = Path(SOURCE)

MAX_COUNT = 1
TEST_ALL = True


def test_process_all(
    transcriber: Callable[[Path, TimeChecker], TRNFormat],
    preprocess: Callable[[Path], Path] = lambda x: x,
    max_count: int = -1,
) -> dict:
    processed_time = TimeChecker()
    transcribe_time = TimeChecker()

    t = lambda x: transcriber(x, transcribe_time)

    processed_time.start()
    data = search_all_ref_and_hyp(src, t, preprocess, max_count)
    processed_time.check()

    concat_result = {}
    for value in data.values():
        for k, v in value.items():
            if k not in concat_result:
                concat_result[k] = []
            concat_result[k].append(v)
    output = sclite_trn(
        concat_result["ref"],
        concat_result["hyp"],
    )
    result = parse_sclite_summary(output)
    result["processed_time"] = processed_time.metric()
    result["transcribe_time"] = transcribe_time.metric()

    return result


def test_process_each(
    transcriber: Callable[[Path], TRNFormat],
    preprocess: Callable[[Path], Path] = lambda x: x,
    max_count: int = -1,
) -> dict:
    processed_time = TimeChecker()
    transcribe_time = TimeChecker()

    t = lambda x: transcriber(x, transcribe_time)

    processed_time.start()
    data = search_all_ref_and_hyp(src, t, preprocess, max_count)
    processed_time.check()

    result = {}
    for key, value in data.items():
        output = sclite_trn([value["ref"]], [value["hyp"]])
        result[key] = parse_sclite_summary(output)
    result["processed_time"] = processed_time.metric()
    result["transcribe_time"] = transcribe_time.metric()
    return result


@lru_cache(maxsize=128)
def load_mp4(mp4, sr=SAMPLE_RATE):
    return load_audio_from_mp4(mp4, sr=sr)


def normalize_text(text: str):
    return normalize_text_only_en(text).upper()


def whisper_streaming():
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    print("Running Whisper Streaming...")

    asr = FasterWhisperASR("en", MODEL_SIZE)
    asr.use_vad()
    online = OnlineASRProcessor(asr)
    rng = np.random.default_rng(42)

    def transcriber(mp4: Path, transcribe_time: TimeChecker) -> TRNFormat:

        audio, _ = load_mp4(mp4, sr=SAMPLE_RATE)
        online.init()

        full_text = ""
        for segment in segment_audio(audio, rng=rng):
            transcribe_time.start()
            online.insert_audio_chunk(segment)
            _, _, text = online.process_iter()
            transcribe_time.check()
            full_text += text
        _, _, text = online.finish()
        full_text += text
        text = normalize_text(full_text)
        return text

    result = (
        test_process_all(transcriber, normalize_text, MAX_COUNT)
        if TEST_ALL
        else test_process_each(transcriber, normalize_text, MAX_COUNT)
    )

    return result


def rt_whisper():
    from rt_whisper import streamers
    from rt_whisper.data import Param, Result


    print("Running RT Whisper...")

    token_streamer = streamers.get_token_streamer_with_vad_v2(
        hyperparameter=HYPERPARAMETER,
    )
    rng = np.random.default_rng(42)

    def transcriber(mp4: Path, transcribe_time: TimeChecker) -> TRNFormat:

        audio, _ = load_mp4(mp4, sr=SAMPLE_RATE)

        completed = []
        param = Param()
        for segment in segment_audio(audio, rng=rng):
            param.chunk = segment
            param.language = "en"
            transcribe_time.start()
            result: Result = token_streamer.process(param)
            transcribe_time.check()
            completed.extend(result.completed)
            param.update(result, update_prompt=True)
        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        text = normalize_text(text)

        return text

    result = (
        test_process_all(transcriber, normalize_text, MAX_COUNT)
        if TEST_ALL
        else test_process_each(transcriber, normalize_text, MAX_COUNT)
    )

    return result


def whisper():
    from faster_whisper import WhisperModel

    print("Running Whisper...")

    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    def transcriber(mp4: Path, transcribe_time: TimeChecker) -> TRNFormat:

        audio, _ = load_mp4(mp4, sr=SAMPLE_RATE)

        transcribe_time.start()
        segments, _ = model.transcribe(
            audio,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            language="en",
            word_timestamps=True,
        )
        transcribe_time.check()

        text = segments_to_text(segments)
        text = normalize_text(text)
        return text

    result = (
        test_process_all(transcriber, normalize_text, MAX_COUNT)
        if TEST_ALL
        else test_process_each(transcriber, normalize_text, MAX_COUNT)
    )

    return result


if __name__ == "__main__":
    import json


    print("Starting performance tests...")

    results = {
        "whisper": whisper(),
        "rt_whisper": rt_whisper(),
        "whisper_streaming": whisper_streaming(),
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Performance tests completed.")
