from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path
from whisper.normalizers import EnglishTextNormalizer

from sj_utils.collection import SafetyDict
from sj_utils.audio import segment_audio

from rt_whisper.models.boundary_word_filter import BoundaryWordFilter

if TYPE_CHECKING:
    pass


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
        for segment in segment_audio(
            audio,
            mean=chunk_size_mean,
            std=chunk_size_std,
            max_div=chunk_size_max_div,
            rng=rng,
        ):
            param.chunk = segment
            param.language = language
            result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result, update_prompt=use_prompt)

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
        for segment in segment_audio(
            audio,
            mean=chunk_size_mean,
            std=chunk_size_std,
            max_div=chunk_size_max_div,
            rng=rng,
        ):
            param.chunk = segment
            param.language = language
            result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result, update_prompt=False)

        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    def token_loader(
        save_path: Path,
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
        for _ in range(segment_length):
            param.language = language
            result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result, update_prompt=False)

        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    def transcriber(
        audio: np.ndarray,
        audio_key: Path | str,
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
                hyperparameter=hyperparameter,
                model=model,
                language=language,
            )
        return token_saver(
            audio,
            saved_path,
            hyperparameter=hyperparameter,
            model=model,
            chunk_size_mean=chunk_size_mean,
            chunk_size_std=chunk_size_std,
            chunk_size_max_div=chunk_size_max_div,
            rng=rng,
            language=language,
        )

    return transcriber


normalizer = EnglishTextNormalizer()


def normalize_text(text: str):
    return normalizer(text)


__all__ = [
    "get_rt_whisper_transcriber",
    "get_token_saver_loader_transcriber",
    "normalize_text",
]
