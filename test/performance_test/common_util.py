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


def test_process_all(
    data_paths: list[Path],
    transcriber: Callable[[np.ndarray, Path, TimeChecker], str],
    generate_ref_and_hyp: Callable[
        [list[Path], Callable[[np.ndarray, Path], str], Callable[[str], str], int],
        tuple[list[TRNFormat], list[TRNFormat]],
    ],
    normalizer: Callable[[Path], Path] = normalize_text,
    max_count: int = -1,
    sr: int = 16000,
) -> dict:
    processed_time = TimeChecker()
    transcribe_time = TimeChecker()

    t = lambda audio, path: transcriber(audio, path, transcribe_time)

    processed_time.start()
    ref, hyp = generate_ref_and_hyp(data_paths, t, normalizer, max_count, sr)
    processed_time.check()

    output = sclite_trn(ref, hyp)
    result = parse_sclite_summary(output)
    result["processed_time"] = processed_time.metric()
    result["transcribe_time"] = transcribe_time.metric()

    return result


def test_process_each(
    data_paths: list[Path],
    transcriber: Callable[[np.ndarray, Path, TimeChecker], str],
    generate_ref_and_hyp: Callable[
        [list[Path], Callable[[np.ndarray, Path], str], Callable[[str], str], int],
        tuple[list[TRNFormat], list[TRNFormat]],
    ],
    normalizer: Callable[[Path], Path] = normalize_text,
    max_count: int = -1,
    sr: int = 16000,
) -> dict:
    processed_time = TimeChecker()
    transcribe_time = TimeChecker()

    t = lambda audio, path: transcriber(audio, path, transcribe_time)

    processed_time.start()
    ref, hyp = generate_ref_and_hyp(data_paths, t, normalizer, max_count, sr)
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
    data_paths: list[Path],
    generate_ref_and_hyp: Callable[
        [list[Path], Callable[[np.ndarray], str], Callable[[str], str], int],
        tuple[list[TRNFormat], list[TRNFormat]],
    ],
    model_size: str = "large-v3",
    seed: int = 42,
    language: str = "en",
    chunk_size: int = 48_000,
    test_all: bool = True,
    max_count: int = -1,
):
    print("Running Whisper Streaming...")

    sr = 16000
    rng = np.random.default_rng(seed)
    t = get_whisper_streaming_transcriber(
        rng,
        model_size=model_size,
        language=language,
        audio_chunk_mean=chunk_size,
    )

    def transcriber(audio: np.ndarray, _: Path, time_checker: TimeChecker) -> str:
        return t(audio, time_checker)

    if test_all:
        result = test_process_all(
            data_paths=data_paths,
            transcriber=transcriber,
            generate_ref_and_hyp=generate_ref_and_hyp,
            normalizer=normalize_text,
            max_count=max_count,
            sr=sr,
        )
    else:
        result = test_process_each(
            data_paths=data_paths,
            transcriber=transcriber,
            generate_ref_and_hyp=generate_ref_and_hyp,
            normalizer=normalize_text,
            max_count=max_count,
            sr=sr,
        )

    del transcriber
    return result


def rt_whisper(
    src: Path,
    storage: Path,
    data_paths: list[Path],
    generate_ref_and_hyp: Callable[
        [list[Path], Callable[[np.ndarray], str], Callable[[str], str], int],
        tuple[list[TRNFormat], list[TRNFormat]],
    ],
    seed: int = 42,
    use_save_loader: bool = True,
    use_prompt: bool = True,
    language: str = "en",
    hyperparameter: Path | SafetyDict = None,
    chunk_size: int = 48_000,
    test_all: bool = True,
    max_count: int = -1,
):
    print("Running RT Whisper...")

    sr = 16000
    if isinstance(hyperparameter, Path):
        _, hyperparameter = load_yaml(hyperparameter)
        hyperparameter = SafetyDict(hyperparameter)
    overlap = hyperparameter["asr"]["max_overlap_duration"]
    rng = np.random.default_rng(seed)

    if use_save_loader:
        t = get_token_saver_loader_transcriber(
            source=src,
            storage=storage,
            overlap=overlap,
            hyperparameter=hyperparameter,
            chunk_size_mean=chunk_size,
            rng=rng,
            language=language,
        )

        def transcriber(
            audio: np.ndarray, path: Path, time_checker: TimeChecker
        ) -> str:
            return t(audio, path, time_checker)

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
        result = test_process_all(
            data_paths=data_paths,
            transcriber=transcriber,
            generate_ref_and_hyp=generate_ref_and_hyp,
            normalizer=normalize_text,
            max_count=max_count,
            sr=sr,
        )
    else:
        result = test_process_each(
            data_paths=data_paths,
            transcriber=transcriber,
            generate_ref_and_hyp=generate_ref_and_hyp,
            normalizer=normalize_text,
            max_count=max_count,
            sr=sr,
        )

    del transcriber
    return result


def whisper(
    data_paths: list[Path],
    generate_ref_and_hyp: Callable[
        [list[Path], Callable[[np.ndarray], str], Callable[[str], str], int],
        tuple[list[TRNFormat], list[TRNFormat]],
    ],
    model_size: str = "large-v3",
    language: str = "en",
    test_all: bool = True,
    max_count: int = -1,
):
    print("Running Whisper...")

    sr = 16000
    t = get_faster_whisper_transcriber(model_size, language)

    def transcriber(audio: np.ndarray, _: Path, time_checker: TimeChecker) -> str:
        return t(audio, time_checker)

    if test_all:
        result = test_process_all(
            data_paths=data_paths,
            transcriber=transcriber,
            generate_ref_and_hyp=generate_ref_and_hyp,
            normalizer=normalize_text,
            max_count=max_count,
            sr=sr,
        )
    else:
        result = test_process_each(
            data_paths=data_paths,
            transcriber=transcriber,
            generate_ref_and_hyp=generate_ref_and_hyp,
            normalizer=normalize_text,
            max_count=max_count,
            sr=sr,
        )

    del transcriber
    return result


def evaluate(
    src: Path,
    storage: Path,
    output_path: Path,
    description: str,
    data_paths: list[Path],
    generate_ref_and_hyp: Callable[
        [list[Path], Callable[[np.ndarray, Path], str], Callable[[str], str], int],
        tuple[list[TRNFormat], list[TRNFormat]],
    ],
    models: list[str] = ["rt_whisper"],
    model_size: str = "large-v3",
    language: str = "en",
    test_all: bool = True,
    max_count: int = -1,
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
                data_paths=data_paths,
                generate_ref_and_hyp=generate_ref_and_hyp,
                model_size=model_size,
                language=language,
                test_all=test_all,
                max_count=max_count,
            )
        elif key == "rt_whisper":
            results[key] = rt_whisper(
                src=src,
                storage=storage,
                data_paths=data_paths,
                generate_ref_and_hyp=generate_ref_and_hyp,
                seed=seed,
                use_save_loader=use_save_loader,
                use_prompt=use_prompt,
                language=language,
                hyperparameter=hyperparameter,
                chunk_size=chunk_size,
                test_all=test_all,
                max_count=max_count,
            )
        elif key == "whisper_streaming":
            results[key] = whisper_streaming(
                data_paths=data_paths,
                generate_ref_and_hyp=generate_ref_and_hyp,
                model_size=model_size,
                seed=seed,
                language=language,
                chunk_size=chunk_size,
                test_all=test_all,
                max_count=max_count,
            )

    json_saver.save(results, output_path)
    print(f"Results saved to {output_path}")


__all__ = [
    "whisper_streaming",
    "rt_whisper",
    "whisper",
    "evaluate",
]
