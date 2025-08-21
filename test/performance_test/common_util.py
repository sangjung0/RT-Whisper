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

from pathlib import Path
from typing import Callable

from sj_utils.file.yaml import load_yaml
from sj_utils.file.json import JsonSaver
from sj_utils.evaluator import TimeChecker
from sj_utils.audio import segment_audio
from sj_utils.collection import SafetyDict
from sj_ai_utils.datasets import Dataset
from sj_ai_utils.asr.whisper_utils import segments_to_text
from sj_ai_utils.evaluator.sclite_utils import (
    TRNFormat,
    sclite_trn,
    parse_sclite_summary,
)

from rt_whisper_optimizer.service import (
    normalize_text,
    get_rt_whisper_transcriber,
    get_token_saver_loader_transcriber,
)


def generate_ref_and_hyp(
    dataset: Dataset,
    transcriber: Callable[[np.ndarray, Path], str],
) -> dict[str, dict[str, list[TRNFormat]]]:
    result_ref = []
    result_hyp = []
    for _id, audio, text in dataset:
        txt = normalize_text(text)
        ref = TRNFormat(id=_id, text=txt)

        pred = transcriber(audio, _id)
        pred = normalize_text(pred)
        hyp = TRNFormat(id=_id, text=pred)

        result_ref.append(ref)
        result_hyp.append(hyp)

    return result_ref, result_hyp


def test_process_all(
    dataset: Dataset,
    transcriber: Callable[[np.ndarray, Path, TimeChecker], str],
) -> dict:
    processed_time = TimeChecker()
    transcribe_time = TimeChecker()

    t = lambda audio, _id: transcriber(audio, _id, transcribe_time)

    processed_time.start()
    ref, hyp = generate_ref_and_hyp(dataset, t)
    processed_time.check()

    output = sclite_trn(ref, hyp)
    result = parse_sclite_summary(output)
    result["processed_time"] = processed_time.metric()
    result["transcribe_time"] = transcribe_time.metric()

    return result


def test_process_each(
    dataset: Dataset,
    transcriber: Callable[[np.ndarray, Path, TimeChecker], str],
) -> dict:
    processed_time = TimeChecker()
    transcribe_time = TimeChecker()

    t = lambda audio, _id: transcriber(audio, _id, transcribe_time)

    processed_time.start()
    ref, hyp = generate_ref_and_hyp(dataset, t)
    processed_time.check()

    result = {}
    for r, h in zip(ref, hyp):
        output = sclite_trn(r, h)
        result[r.id] = parse_sclite_summary(output)

    result["processed_time"] = processed_time.metric()
    result["transcribe_time"] = transcribe_time.metric()
    return result


def get_whisper_streaming_transcriber(
    rng: np.random.Generator | np.random.RandomState = np.random,
    model_size: str = "large-v3",
    language: str = "en",
    audio_chunk_mean: int = 48_000,
    audio_chunk_std: int = 0,
    audio_chunk_max_div: int = 0,
):
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    _audio_chunk_mean = audio_chunk_mean
    _audio_chunk_std = audio_chunk_std
    _audio_chunk_max_div = audio_chunk_max_div

    asr = FasterWhisperASR(language, model_size)
    asr.use_vad()
    online = OnlineASRProcessor(asr)

    def transcriber(
        audio: np.ndarray,
        transcribe_time: TimeChecker,
        audio_chunk_mean: int = _audio_chunk_mean,
        audio_chunk_std: int = _audio_chunk_std,
        audio_chunk_max_div: int = _audio_chunk_max_div,
    ) -> str:
        online.init()
        full_text = ""
        for segment in segment_audio(
            audio,
            mean=audio_chunk_mean,
            std=audio_chunk_std,
            max_div=audio_chunk_max_div,
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


def get_faster_whisper_transcriber(model_size: str = "large-v3", language: str = "en"):
    from faster_whisper import WhisperModel

    _language = language

    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    def transcriber(
        audio: np.ndarray, transcribe_time: TimeChecker, language: str = _language
    ) -> str:
        transcribe_time.start()
        segments, _ = model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
        )
        transcribe_time.check()

        return segments_to_text(segments)

    return transcriber


def whisper_streaming(
    dataset: Dataset,
    model_size: str = "large-v3",
    seed: int = 42,
    language: str = "en",
    chunk_size: int = 48_000,
    test_all: bool = True,
):
    print("Running Whisper Streaming...")

    rng = np.random.default_rng(seed)
    t = get_whisper_streaming_transcriber(
        rng,
        model_size=model_size,
        language=language,
        audio_chunk_mean=chunk_size,
    )

    def transcriber(audio: np.ndarray, _: str, time_checker: TimeChecker) -> str:
        return t(audio, time_checker)

    if test_all:
        result = test_process_all(dataset=dataset, transcriber=transcriber)
    else:
        result = test_process_each(dataset=dataset, transcriber=transcriber)

    del transcriber
    del t
    return result


def rt_whisper(
    storage: Path,
    dataset: Dataset,
    seed: int = 42,
    use_save_loader: bool = True,
    use_prompt: bool = True,
    language: str = "en",
    hyperparameter: Path | SafetyDict = None,
    chunk_size: int = 48_000,
    test_all: bool = True,
):
    print("Running RT Whisper...")

    if isinstance(hyperparameter, Path):
        _, hyperparameter = load_yaml(hyperparameter)
        hyperparameter = SafetyDict(hyperparameter)
    overlap = hyperparameter["asr"]["max_overlap_duration"]
    rng = np.random.default_rng(seed)

    if use_save_loader:
        t = get_token_saver_loader_transcriber(
            storage=storage,
            overlap=overlap,
            hyperparameter=hyperparameter,
            chunk_size_mean=chunk_size,
            rng=rng,
            language=language,
        )

        def transcriber(audio: np.ndarray, _id: str, time_checker: TimeChecker) -> str:
            return t(audio, _id, time_checker)

    else:
        t = get_rt_whisper_transcriber(
            hyperparameter=hyperparameter,
            chunk_size_mean=chunk_size,
            rng=rng,
            language=language,
            use_prompt=use_prompt,
        )

        def transcriber(audio: np.ndarray, _: Path, time_checker: TimeChecker) -> str:
            return t(audio, time_checker)

    if test_all:
        result = test_process_all(dataset=dataset, transcriber=transcriber)
    else:
        result = test_process_each(dataset=dataset, transcriber=transcriber)

    del transcriber
    del t
    return result


def whisper(
    dataset: Dataset,
    model_size: str = "large-v3",
    language: str = "en",
    test_all: bool = True,
):
    print("Running Whisper...")

    t = get_faster_whisper_transcriber(model_size, language)

    def transcriber(audio: np.ndarray, _: str, time_checker: TimeChecker) -> str:
        return t(audio, time_checker)

    if test_all:
        result = test_process_all(dataset=dataset, transcriber=transcriber)
    else:
        result = test_process_each(dataset=dataset, transcriber=transcriber)

    del transcriber
    del t
    return result


def evaluate(
    storage: Path,
    output_path: Path,
    description: str,
    dataset: Dataset,
    models: list[str] = ["rt_whisper"],
    model_size: str = "large-v3",
    language: str = "en",
    test_all: bool = True,
    seed: int = 42,
    use_save_loader: bool = True,
    use_prompt: bool = False,
    hyperparameter: Path = None,
    chunk_size: int = 48_000,
):
    json_saver = JsonSaver(description)

    results = {}
    for key in models:
        if key == "whisper":
            results[key] = whisper(
                dataset=dataset,
                model_size=model_size,
                language=language,
                test_all=test_all,
            )
        elif key == "rt_whisper":
            results[key] = rt_whisper(
                storage=storage,
                dataset=dataset,
                seed=seed,
                use_save_loader=use_save_loader,
                use_prompt=use_prompt,
                language=language,
                hyperparameter=hyperparameter,
                chunk_size=chunk_size,
                test_all=test_all,
            )
        elif key == "whisper_streaming":
            results[key] = whisper_streaming(
                dataset=dataset,
                model_size=model_size,
                seed=seed,
                language=language,
                chunk_size=chunk_size,
                test_all=test_all,
            )

    json_saver.save(results, output_path)
    print(f"Results saved to {output_path}")


__all__ = [
    "whisper_streaming",
    "rt_whisper",
    "whisper",
    "evaluate",
]
