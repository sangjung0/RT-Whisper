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

import time
import librosa
import numpy as np

from pathlib import Path
from typing import Callable

from sj_ai_utils.asr.whisper_utils import *
from sj_ai_utils.datasets.libri_speech_asr_corpus import *
from sj_ai_utils.evaluator.sclite_utils import *
from sj_utils.audio_utils import *
from sj_utils.string_utils import *

MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000
SOURCE = "/workspaces/dev/datasets/LibriSpeechASRcorpus/test/test-clean/"

src = Path(SOURCE)


def test_process(
    transcriber: Callable[[Path], TRNFormat],
    preprocess: Callable[[Path], Path] = lambda x: x,
) -> dict:
    data = search_all_ref_and_hyp(src, transcriber, preprocess, 2)
    concat_result = {}
    for value in data.values():
        for k, v in value.items():
            if k not in concat_result:
                concat_result[k] = []
            concat_result[k].extend(v)
    output = sclite_trn(
        concat_result["ref"],
        concat_result["hyp"],
    )
    return parse_sclite_summary(output)


def whisper_streaming():
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    print("Running Whisper Streaming...")

    asr = FasterWhisperASR("en", MODEL_SIZE)
    asr.use_vad()
    online = OnlineASRProcessor(asr)

    rng = np.random.default_rng(42)
    transcribe_time = 0

    def transcriber(flac: Path) -> TRNFormat:
        nonlocal transcribe_time

        audio, _ = librosa.load(flac, sr=SAMPLE_RATE)
        online.init()

        full_text = ""
        for segment in segment_audio(audio, rng=rng):
            start_time = time.perf_counter()
            online.insert_audio_chunk(segment)
            _, _, text = online.process_iter()
            transcribe_time += time.perf_counter() - start_time
            full_text += text
        _, _, text = online.finish()
        full_text += text

        return TRNFormat(id=flac.stem, text=normalize_text_only_en(full_text).upper())

    start_time = time.perf_counter()
    result = test_process(transcriber, lambda x: normalize_text_only_en(x).upper())
    result["processed_time"] = time.perf_counter() - start_time
    result["transcribe_time"] = transcribe_time
    return result


def rt_whisper():
    from rt_whisper import streamers
    from rt_whisper.data import Param, Result

    HYPERPARAMETER = "./hyperparameters/sclite.yml"

    print("Running RT Whisper...")

    token_streamer = streamers.get_token_streamer_with_vad_v2(
        hyperparameter=HYPERPARAMETER,
    )

    rng = np.random.default_rng(42)
    transcribe_time = 0

    def transcriber(flac: Path) -> TRNFormat:
        nonlocal transcribe_time

        audio, _ = librosa.load(flac, sr=SAMPLE_RATE)

        completed = []
        param = Param()

        for segment in segment_audio(audio, rng=rng):
            param.chunk = segment
            param.language = "en"
            start_time = time.perf_counter()
            result: Result = token_streamer.process(param)
            transcribe_time += time.perf_counter() - start_time
            completed.extend(result.completed)
            param.update(result, update_prompt=True)
        completed.extend(result.candidate)

        return TRNFormat(
            id=flac.stem,
            text=normalize_text_only_en(" ".join([s.text for s in completed])).upper(),
        )

    start_time = time.perf_counter()
    result = test_process(transcriber, lambda x: normalize_text_only_en(x).upper())
    result["processed_time"] = time.perf_counter() - start_time
    result["transcribe_time"] = transcribe_time
    return result


def whisper():
    from faster_whisper import WhisperModel

    print("Running Whisper...")

    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    transcribe_time = 0

    def transcriber(flac: Path) -> TRNFormat:
        nonlocal transcribe_time

        audio, _ = librosa.load(flac, sr=SAMPLE_RATE)

        start_time = time.perf_counter()
        segments, _ = model.transcribe(
            audio,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            language="en",
            word_timestamps=True,
        )
        transcribe_time += time.perf_counter() - start_time

        trn = segments_to_sclite_trn(flac.stem, segments)
        trn.text = normalize_text_only_en(trn.text).upper()
        return trn

    start_time = time.perf_counter()
    result = test_process(
        transcriber,
        lambda x: normalize_text_only_en(x).upper(),
    )
    result["processed_time"] = time.perf_counter() - start_time
    result["transcribe_time"] = transcribe_time
    return result


if __name__ == "__main__":
    import json

    OUTPUT_PATH = "/workspaces/dev/output/result.json"

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
