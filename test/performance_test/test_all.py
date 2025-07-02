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

from pathlib import Path
import librosa
from typing import Callable

from sj_ai_utils.asr.whisper_utils import *
from sj_ai_utils.datasets.libri_speech_asr_corpus import *
from sj_ai_utils.evaluator.sclite_utils import *
from sj_utils.audio_utils import *
from sj_utils.string_utils import *

MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000
OUTPUT_PATH = "/workspaces/dev/output/result.json"
SOURCE = (
    "/workspaces/dev/datasets/LibriSpeechASRcorpus/test-other/LibriSpeech/test-other/"
)
DESTINATION = {
    "whisper": "/workspaces/dev/output/LibriSpeechASRcorpus/sclient/whisper/test-other/",
    "rt_whisper": "/workspaces/dev/output/LibriSpeechASRcorpus/sclient/rt_whisper/test-other/",
    "whisper_streaming": "/workspaces/dev/output/LibriSpeechASRcorpus/sclient/whisper_streaming/test-other/",
}

# only rt whisper


src = Path(SOURCE)
dest = {key: Path(value) for key, value in DESTINATION.items()}
for value in dest.values():
    value.mkdir(parents=True, exist_ok=True)


def test_process(dest_path: Path, transcriber: Callable[[Path], TRNFormat]) -> dict:
    make_all_ref_and_hyp(src, dest_path, transcriber, 1)
    concat_trn_file(
        list(sorted(p for p in dest_path.rglob("*.ref.trn"))),
        dest_path / "concat.ref.trn",
    )
    concat_trn_file(
        list(sorted(p for p in dest_path.rglob("*.hyp.trn"))),
        dest_path / "concat.hyp.trn",
    )

    output = sclite_trn_run(
        dest_path / "concat.ref.trn",
        dest_path / "concat.hyp.trn",
    )

    return parse_sclite_summary(output)


def whisper_streaming():
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    print("Running Whisper Streaming...")

    asr = FasterWhisperASR("en", MODEL_SIZE)
    asr.use_vad()
    online = OnlineASRProcessor(asr)

    def transcriber(flac: Path) -> TRNFormat:
        audio, _ = librosa.load(flac, sr=SAMPLE_RATE)
        online.init()

        full_text = ""
        for segment in segment_audio(audio):
            online.insert_audio_chunk(segment)
            _, _, text = online.process_iter()
            full_text += text
        _, _, text = online.finish()
        full_text += text

        return TRNFormat(id=flac.stem, text=normalize_text_only_en(full_text).upper())

    return test_process(dest["whisper_streaming"], transcriber)


def rt_whisper():
    from rt_whisper import streamers
    from rt_whisper.data import Param, Result

    HYPERPARAMETER_PATH = "./hyperparameters/sclite.yml"

    print("Running RT Whisper...")

    token_streamer = streamers.get_token_streamer_with_vad_v2(
        hyperparameter_path=HYPERPARAMETER_PATH,
    )

    def transcriber(flac: Path) -> TRNFormat:
        audio, _ = librosa.load(flac, sr=SAMPLE_RATE)

        completed = []
        param = Param()

        for segment in segment_audio(audio):
            param.chunk = segment
            param.language = "en"
            result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result)
        completed.extend(result.candidate)

        return TRNFormat(
            id=flac.stem,
            text=normalize_text_only_en(" ".join([s.text for s in completed])).upper(),
        )

    return test_process(dest["rt_whisper"], transcriber)


def whisper():
    from faster_whisper import WhisperModel

    print("Running Whisper...")

    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    def transcriber(flac: Path) -> TRNFormat:
        audio, _ = librosa.load(flac, sr=SAMPLE_RATE)
        segments, _ = model.transcribe(
            audio,
            beam_size=5,
            language="en",
            word_timestamps=True,
        )

        trn = segments_to_sclite_trn(flac.stem, segments)
        trn.text = normalize_text_only_en(trn.text).upper()
        return trn

    return test_process(dest["whisper"], transcriber)


if __name__ == "__main__":
    import json

    print("Starting performance tests...")

    results = {
        "whisper": whisper(),
        "rt_whisper": rt_whisper(),
        "whisper_streaming": whisper_streaming(),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Performance tests completed.")
