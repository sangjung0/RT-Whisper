import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/test/modules/whisper_streaming",
]
for path in paths:
    sys.path.append(os.path.abspath(path))

import numpy as np

from pathlib import Path
from typing import Callable
from whisper.normalizers import EnglishTextNormalizer

from sj_utils.file.yaml import load_yaml
from sj_utils.file.json import JsonSaver
from sj_utils.evaluator import TimeChecker
from sj_utils.evaluator.asr import TimeEvaluator, TimeEvaluatorSummary
from sj_utils.audio import segment_audio
from sj_utils.collection import SafetyDict
from sj_ai_utils.datasets import Dataset
from sj_ai_utils.asr.whisper_utils import segments_to_text
from sj_ai_utils.evaluator.sclite_utils import (
    TRNFormat,
    sclite_trn,
    parse_sclite_summary,
)

from rt_whisper.models.boundary_word_filter import BoundaryWordFilter

normalizer = EnglishTextNormalizer()


def normalize_text(text: str):
    return normalizer(text)


def test_process(
    dataset: Dataset,
    transcriber: Callable[[np.ndarray, Path, TimeEvaluator], str],
    all: bool,
) -> dict:
    result_ref = []
    result_hyp = []
    processed_time = TimeChecker()
    transcribe_time = TimeEvaluatorSummary()

    with processed_time.timeit():
        for _id, audio, text in dataset:
            te = TimeEvaluator(L=len(audio))

            txt = normalize_text(text)
            ref = TRNFormat(id=_id, text=txt)

            pred = transcriber(audio, _id, te)
            pred = normalize_text(pred)
            hyp = TRNFormat(id=_id, text=pred)

            result_ref.append(ref)
            result_hyp.append(hyp)
            transcribe_time.add(te)

    if all:
        output = sclite_trn(result_ref, result_hyp)
        result = parse_sclite_summary(output)
    else:
        result = {}
        for r, h in zip(result_ref, result_hyp):
            output = sclite_trn([r], [h])
            result[r.id] = parse_sclite_summary(output)

    result["processed_time"] = processed_time.metric()
    result["transcribe_time"] = transcribe_time.metric()

    return result


def test_process_for_rt(
    dataset: Dataset,
    transcriber: Callable[[np.ndarray, Path, TimeEvaluator, TimeEvaluator], str],
    all: bool,
) -> dict:
    result_ref = []
    result_hyp = []
    processed_time = TimeChecker()
    completed = TimeEvaluatorSummary()
    candidate = TimeEvaluatorSummary()

    with processed_time.timeit():
        for _id, audio, text in dataset:
            te = TimeEvaluator(L=len(audio))
            te2 = TimeEvaluator(L=len(audio))

            txt = normalize_text(text)
            ref = TRNFormat(id=_id, text=txt)

            pred = transcriber(audio, _id, te, te2)
            pred = normalize_text(pred)
            hyp = TRNFormat(id=_id, text=pred)

            result_ref.append(ref)
            result_hyp.append(hyp)
            completed.add(te)
            candidate.add(te2)

    if all:
        output = sclite_trn(result_ref, result_hyp)
        result = parse_sclite_summary(output)
    else:
        result = {}
        for r, h in zip(result_ref, result_hyp):
            output = sclite_trn([r], [h])
            result[r.id] = parse_sclite_summary(output)

    result["processed_time"] = processed_time.metric()
    result["completed_time"] = completed.metric()
    result["candidate_time"] = candidate.metric()

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
        transcribe_time: TimeEvaluator,
        audio_chunk_mean: int = _audio_chunk_mean,
        audio_chunk_std: int = _audio_chunk_std,
        audio_chunk_max_div: int = _audio_chunk_max_div,
    ) -> str:
        online.init()
        full_text = ""
        input_length = 0
        for segment in segment_audio(
            audio,
            mean=audio_chunk_mean,
            std=audio_chunk_std,
            max_div=audio_chunk_max_div,
            rng=rng,
        ):
            input_length += len(segment)
            with transcribe_time.timeit():
                online.insert_audio_chunk(segment)
                _, _, text = online.process_iter()
            full_text += text
            transcribe_time.add_coverage(
                np.full(len(text.split()), input_length, dtype=np.float32)
            )
        _, _, text = online.finish()
        full_text += text
        transcribe_time.add_coverage(
            np.full(len(text.split()), input_length, dtype=np.float32)
        )
        return full_text

    return transcriber


def get_faster_whisper_transcriber(model_size: str = "large-v3", language: str = "en"):
    from faster_whisper import WhisperModel

    _language = language

    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    def transcriber(
        audio: np.ndarray, transcribe_time: TimeEvaluator, language: str = _language
    ) -> str:
        with transcribe_time.timeit():
            segments, _ = model.transcribe(
                audio,
                language=language,
                word_timestamps=True,
            )
            text = segments_to_text(segments)
        transcribe_time.add_coverage(
            np.full(len(text.split()), len(audio), dtype=np.float32)
        )

        return text

    return transcriber


def get_rt_whisper_transcriber(
    hyperparameter: SafetyDict | Path = None,
    chunk_size_mean: int = 48_000,
    chunk_size_std: int = 0,
    chunk_size_max_div: int = 0,
    rng: np.random.Generator | np.random.RandomState = np.random,
    language: str = "en",
    use_prompt: bool = True,
):
    from rt_whisper.data import Param, Result
    from rt_whisper.streamers import get_token_streamer_with_vad_v2_min_filter

    _hyperparameter = hyperparameter
    _chunk_size_mean = chunk_size_mean
    _chunk_size_std = chunk_size_std
    _chunk_size_max_div = chunk_size_max_div
    _rng = rng
    _language = language
    _use_prompt = use_prompt

    def transcriber(
        audio: np.ndarray,
        completed_time: TimeEvaluator,
        candidate_time: TimeEvaluator,
        hyperparameter: SafetyDict | Path = _hyperparameter,
        model: BoundaryWordFilter = None,
        chunk_size_mean: int = _chunk_size_mean,
        chunk_size_std: int = _chunk_size_std,
        chunk_size_max_div: int = _chunk_size_max_div,
        rng: np.random.Generator | np.random.RandomState = _rng,
        language: str = _language,
        use_prompt: bool = _use_prompt,
    ) -> str:
        token_streamer = get_token_streamer_with_vad_v2_min_filter(
            hyperparameter=hyperparameter
        )

        if model is not None:
            token_streamer._Pipeline__workers[1][0].model.model = model

        completed = []
        param = Param()
        input_length = 0
        end = 0
        for segment in segment_audio(
            audio,
            mean=chunk_size_mean,
            std=chunk_size_std,
            max_div=chunk_size_max_div,
            rng=rng,
        ):
            input_length += len(segment)
            param.chunk = segment
            param.language = language
            with completed_time.timeit():
                result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result, update_prompt=use_prompt)

            completed_tokens = [t for t in result.completed_tokens if t.is_word]
            candidate_tokens = [
                t for t in result.candidate_tokens if t.is_word and t.start >= end
            ]
            end = candidate_tokens[-1].end if candidate_tokens else end
            completed_time.add_coverage(
                np.full(len(completed_tokens), input_length, dtype=np.float32)
            )
            candidate_time.add_coverage(
                np.full(len(candidate_tokens), input_length, dtype=np.float32)
            )

        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    return transcriber


def get_token_saver_loader_transcriber(
    storage: Path,
    overlap: int = None,
    hyperparameter: SafetyDict = None,
    chunk_size_mean: int = 48_000,
    chunk_size_std: int = 0,
    chunk_size_max_div: int = 0,
    rng: np.random.Generator | np.random.RandomState = np.random,
    language: str = "en",
):
    from rt_whisper import saveloaders
    from rt_whisper.data import Param, Result

    _storage = storage
    _overlap = overlap
    _hyperparameter = hyperparameter
    _chunk_size_mean = chunk_size_mean
    _chunk_size_std = chunk_size_std
    _chunk_size_max_div = chunk_size_max_div
    _rng = rng
    _language = language

    def token_saver(
        audio: np.ndarray,
        save_path: Path,
        completed_time: TimeEvaluator,
        candidate_time: TimeEvaluator,
        hyperparameter: SafetyDict = _hyperparameter,
        model: BoundaryWordFilter = None,
        chunk_size_mean: int = _chunk_size_mean,
        chunk_size_std: int = _chunk_size_std,
        chunk_size_max_div: int = _chunk_size_max_div,
        rng: np.random.Generator | np.random.RandomState = _rng,
        language: str = _language,
    ) -> str:
        token_streamer = saveloaders.get_token_streamer_saver(
            save_path=save_path, hyperparameter=hyperparameter
        )
        if model is not None:
            token_streamer._Pipeline__workers[2][0].model.model = model

        completed = []
        param = Param()
        input_length = 0
        end = 0
        for segment in segment_audio(
            audio,
            mean=chunk_size_mean,
            std=chunk_size_std,
            max_div=chunk_size_max_div,
            rng=rng,
        ):
            input_length += len(segment)
            param.chunk = segment
            param.language = language
            with completed_time.timeit():
                result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result, update_prompt=False)

            completed_tokens = [t for t in result.completed_tokens if t.is_word]
            candidate_tokens = [
                t for t in result.candidate_tokens if t.is_word and t.start >= end
            ]
            end = candidate_tokens[-1].end if candidate_tokens else end
            completed_time.add_coverage(
                np.full(len(completed_tokens), input_length, dtype=np.float32)
            )
            candidate_time.add_coverage(
                np.full(len(candidate_tokens), input_length, dtype=np.float32)
            )

        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    def token_loader(
        save_path: Path,
        completed_time: TimeEvaluator,
        candidate_time: TimeEvaluator,
        hyperparameter: SafetyDict = _hyperparameter,
        model: BoundaryWordFilter = None,
        language: str = _language,
    ) -> str:
        token_streamer = saveloaders.get_token_streamer_loader(
            saved_path=save_path, hyperparameter=hyperparameter
        )
        if model is not None:
            token_streamer._Pipeline__workers[1][0].model.model = model

        segment_length = len(list(save_path.iterdir()))

        completed = []
        param = Param()
        input_length = 0
        end = 0
        for _ in range(segment_length):
            input_length += 1
            param.language = language
            with completed_time.timeit():
                result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result, update_prompt=False)

            completed_tokens = [t for t in result.completed_tokens if t.is_word]
            candidate_tokens = [
                t for t in result.candidate_tokens if t.is_word and t.start >= end
            ]
            end = candidate_tokens[-1].end if candidate_tokens else end
            completed_time.add_coverage(
                np.full(len(completed_tokens), input_length, dtype=np.float32)
            )
            candidate_time.add_coverage(
                np.full(len(candidate_tokens), input_length, dtype=np.float32)
            )

        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    def transcriber(
        audio: np.ndarray,
        audio_key: Path | str,
        completed_time: TimeEvaluator,
        candidate_time: TimeEvaluator,
        storage: Path = _storage,
        overlap: int = _overlap,
        hyperparameter: SafetyDict = _hyperparameter,
        model: BoundaryWordFilter = None,
        chunk_size_mean: int = _chunk_size_mean,
        chunk_size_std: int = _chunk_size_std,
        chunk_size_max_div: int = _chunk_size_max_div,
        rng: np.random.Generator | np.random.RandomState = _rng,
        language: str = _language,
    ) -> str:
        if hyperparameter is None:
            raise ValueError("hyperparameter must be provided")
        if overlap is None:
            raise ValueError("overlap must be provided")

        saved_path = storage / f"{overlap}" / audio_key

        if saved_path.exists():
            return token_loader(
                saved_path,
                completed_time,
                candidate_time,
                hyperparameter=hyperparameter,
                model=model,
                language=language,
            )
        return token_saver(
            audio,
            saved_path,
            completed_time,
            candidate_time,
            hyperparameter=hyperparameter,
            model=model,
            chunk_size_mean=chunk_size_mean,
            chunk_size_std=chunk_size_std,
            chunk_size_max_div=chunk_size_max_div,
            rng=rng,
            language=language,
        )

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

    result = test_process(dataset=dataset, transcriber=transcriber, all=test_all)

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
        transcriber = get_token_saver_loader_transcriber(
            storage=storage,
            overlap=overlap,
            hyperparameter=hyperparameter,
            chunk_size_mean=chunk_size,
            rng=rng,
            language=language,
        )

    else:
        t = get_rt_whisper_transcriber(
            hyperparameter=hyperparameter,
            chunk_size_mean=chunk_size,
            rng=rng,
            language=language,
            use_prompt=use_prompt,
        )

        def transcriber(audio: np.ndarray, _: Path, completed_time: TimeChecker, candidate_time: TimeChecker) -> str:
            return t(audio, completed_time, candidate_time)

    result = test_process_for_rt(dataset=dataset, transcriber=transcriber, all=test_all)

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

    result = test_process(dataset=dataset, transcriber=transcriber, all=test_all)

    del transcriber
    del t
    return result


def evaluate(
    output_path: Path,
    description: str,
    dataset: Dataset,
    storage: Path = None,
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
