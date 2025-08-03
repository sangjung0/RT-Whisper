import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/modules/python-utils",
    "/workspaces/dev/modules/ai-utils",
    "/workspaces/dev",
    "/workspaces/dev/test/modules/whisper_streaming",
]
for path in paths:
    sys.path.append(os.path.abspath(path))

import numpy as np
import string

from pathlib import Path
from typing import Callable
from whisper.normalizers import EnglishTextNormalizer

from sj_ai_utils.asr.whisper_utils import segments_to_text
from sj_utils.evaluator import TimeChecker
from sj_ai_utils.evaluator.sclite_utils import (
    TRNFormat,
    sclite_trn,
    parse_sclite_summary,
)
from sj_utils.audio import segment_audio
from sj_utils.collection import SafetyDict

SAMPLE_RATE = 16_000
MODEL_SIZE = "large-v3"


normalizer = EnglishTextNormalizer()


def normalize_text(text: str):
    text = normalizer(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def test_process_all(
    data_paths: list[Path],
    transcriber: Callable[[Path, TimeChecker], str],
    make_ref_and_hyp: Callable[
        [list[Path], Callable[[Path], str], Callable[[str], str], int, bool],
        dict[str, dict[str, list[TRNFormat]]],
    ],
    normalizer: Callable[[Path], Path] = normalize_text,
    max_count: int = -1,
) -> dict:
    processed_time = TimeChecker()
    transcribe_time = TimeChecker()

    t = lambda x: transcriber(x, transcribe_time)

    processed_time.start()
    data = make_ref_and_hyp(data_paths, t, normalizer, max_count)
    processed_time.check()

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
    result = parse_sclite_summary(output)
    result["processed_time"] = processed_time.metric()
    result["transcribe_time"] = transcribe_time.metric()

    return result


def test_process_each(
    data_paths: list[Path],
    transcriber: Callable[[Path], str],
    make_ref_and_hyp: Callable[
        [list[Path], Callable[[Path], str], Callable[[str], str], int, bool],
        dict[str, dict[str, list[TRNFormat]]],
    ],
    normalizer: Callable[[Path], Path] = normalize_text,
    max_count: int = -1,
) -> dict:
    processed_time = TimeChecker()
    transcribe_time = TimeChecker()

    t = lambda x: transcriber(x, transcribe_time)

    processed_time.start()
    data = make_ref_and_hyp(data_paths, t, normalizer, max_count)
    processed_time.check()

    result = {}
    for key, value in data.items():
        output = sclite_trn(value["ref"], value["hyp"])
        result[key] = parse_sclite_summary(output)
    result["processed_time"] = processed_time.metric()
    result["transcribe_time"] = transcribe_time.metric()
    return result


def get_whisper_streaming_transcriber(
    load_audio: Callable[[Path, int], tuple[np.ndarray, int]],
    rng: np.random.Generator | np.random.RandomState = np.random,
    sr: int = SAMPLE_RATE,
    audio_chunk_mean: int = 48000,
    audio_chunk_std: int = 400,
    audio_chunk_min_max: float = 0.1,
):
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    asr = FasterWhisperASR("en", MODEL_SIZE)
    asr.use_vad()
    online = OnlineASRProcessor(asr)

    def transcriber(src: Path, transcribe_time: TimeChecker) -> str:

        audio, _ = load_audio(src, sr=sr)
        online.init()

        full_text = ""
        for segment in segment_audio(
            audio,
            mean=audio_chunk_mean,
            std=audio_chunk_std,
            ratio=audio_chunk_min_max,
            rng=rng,
        ):
            transcribe_time.start()
            online.insert_audio_chunk(segment)
            _, _, text = online.process_iter()
            transcribe_time.check()
            full_text += text
        _, _, text = online.finish()
        full_text += text
        return full_text

    return transcriber


def get_rt_whisper_transcriber(
    hyperparameter: SafetyDict,
    load_audio: Callable[[Path, int], tuple[np.ndarray, int]],
    rng: np.random.Generator | np.random.RandomState = np.random,
    sr: int = SAMPLE_RATE,
    audio_chunk_mean: int = 48000,
    audio_chunk_std: int = 400,
    audio_chunk_min_max: float = 0.1,
):
    from rt_whisper.data import Param, Result
    from rt_whisper import streamers

    token_streamer = streamers.get_token_streamer_with_vad_v2_min_filter(hyperparameter)

    def transcriber(src: Path, transcribe_time: TimeChecker) -> str:
        audio, _ = load_audio(src, sr=sr)

        completed = []
        param = Param()
        for segment in segment_audio(
            audio,
            mean=audio_chunk_mean,
            std=audio_chunk_std,
            ratio=audio_chunk_min_max,
            rng=rng,
        ):
            param.chunk = segment
            param.language = "en"
            transcribe_time.start()
            result: Result = token_streamer.process(param)
            transcribe_time.check()
            completed.extend(result.completed)
            param.update(result, update_prompt=True)
        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    return transcriber


def get_faster_whisper_transcriber(
    load_audio: Callable[[Path, int], tuple[np.ndarray, int]],
    sr: int = SAMPLE_RATE,
):
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    def transcriber(src: Path, transcribe_time: TimeChecker) -> str:

        audio, _ = load_audio(src, sr=sr)

        transcribe_time.start()
        segments, _ = model.transcribe(
            audio,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            language="en",
            word_timestamps=True,
        )
        transcribe_time.check()

        return segments_to_text(segments)

    return transcriber


def get_token_saver_loader_transcriber(
    source: Path,
    storage: Path,
    load_audio: Callable[[Path, int], tuple[np.ndarray, int]],
    rng: np.random.Generator | np.random.RandomState = np.random,
    hyperparameter: SafetyDict = None,
    overlap: int = None,
    sr: int = SAMPLE_RATE,
):
    from rt_whisper import saveloaders
    from rt_whisper.data import Param, Result

    saved_hyperparameter = hyperparameter
    saved_overlap = overlap

    def token_saver(
        audio_src: Path,
        save_path: Path,
        hyperparameter: SafetyDict,
        transcribe_time: TimeChecker,
    ) -> str:
        token_streamer = saveloaders.get_token_streamer_saver(
            save_path=save_path, hyperparameter=hyperparameter
        )
        return get_rt_whisper_transcriber(token_streamer, sr, load_audio, rng)(
            audio_src, transcribe_time
        )

    def token_loader(
        saved_path: Path,
        hyperparameter: SafetyDict,
        transcribe_time: TimeChecker,
    ) -> str:
        token_streamer = saveloaders.get_token_streamer_loader(
            saved_path=saved_path, hyperparameter=hyperparameter
        )

        segment_length = len(list(saved_path.iterdir()))

        completed = []
        param = Param()
        for _ in range(segment_length):
            param.language = "en"
            transcribe_time.start()
            result: Result = token_streamer.process(param)
            transcribe_time.check()
            completed.extend(result.completed)
            param.update(result, update_prompt=True)
        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    def transcriber(
        audio_src: Path,
        transcribe_time: TimeChecker,
        hyperparameter: SafetyDict = saved_hyperparameter,
        overlap: int = saved_overlap,
    ) -> str:
        if hyperparameter is None:
            raise ValueError("hyperparameter must be provided")
        if overlap is None:
            raise ValueError("overlap must be provided")

        relative_path = audio_src.parent.relative_to(source.parent) / audio_src.stem
        saved_path = storage / f"{overlap}" / relative_path

        if saved_path.exists():
            return token_loader(saved_path, hyperparameter, transcribe_time)
        return token_saver(audio_src, saved_path, hyperparameter, transcribe_time)

    return transcriber


__all__ = [
    "test_process_all",
    "test_process_each",
    "normalize_text",
    "get_whisper_streaming_transcriber",
    "get_rt_whisper_transcriber",
    "get_faster_whisper_transcriber",
    "get_token_saver_loader_transcriber",
]
