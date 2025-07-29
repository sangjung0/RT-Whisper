from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path
from typing import Callable
from whisper.normalizers import EnglishTextNormalizer

from sj_utils.evaluator import TimeChecker
from sj_utils.collection import SafetyDict
from sj_utils.audio import segment_audio

if TYPE_CHECKING:
    pass


def get_rt_whisper_transcriber(
    sr: int,
    load_audio: Callable[[Path, int], tuple[np.ndarray, int]],
    use_prompt: bool,
    rng: np.random.Generator | np.random.RandomState = np.random,
):
    from rt_whisper.data import Param, Result
    from rt_whisper.streamers import get_token_streamer_with_vad_v2_min_filter

    def transcriber(
        src: Path, transcribe_time: TimeChecker, hyperparameter: SafetyDict | Path
    ) -> str:
        token_streamer = get_token_streamer_with_vad_v2_min_filter(
            hyperparameter=hyperparameter
        )

        audio, _ = load_audio(src, sr=sr)

        completed = []
        param = Param()
        for segment in segment_audio(audio, rng=rng):
            param.chunk = segment
            param.language = "en"
            transcribe_time.start()
            result: Result = token_streamer.process(param)
            transcribe_time.check()
            completed.extend(result.completed)
            param.update(result, update_prompt=use_prompt)
        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    return transcriber


def get_token_saver_loader_transcriber(
    source: Path,
    storage: Path,
    sr: int,
    load_audio: Callable[[Path, int], tuple[np.ndarray, int]],
    rng: np.random.Generator | np.random.RandomState = np.random,
    hyperparameter: SafetyDict = None,
    overlap: int = None,
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

        audio, _ = load_audio(audio_src, sr=sr)

        completed = []
        param = Param()
        for segment in segment_audio(audio, rng=rng):
            param.chunk = segment
            param.language = "en"
            transcribe_time.start()
            result: Result = token_streamer.process(param)
            transcribe_time.check()
            completed.extend(result.completed)
            param.update(result, update_prompt=False)
        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

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


normalizer = EnglishTextNormalizer()


def normalize_text(text: str):
    return normalizer(text)


__all__ = [
    "get_rt_whisper_transcriber",
    "get_token_saver_loader_transcriber",
    "normalize_text",
]
