from __future__ import annotations
from typing import TYPE_CHECKING

import os
import time
import joblib
import librosa
import random
import optuna
import copy
import optuna.visualization as vis

import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable
from functools import lru_cache

from sj_ai_utils.evaluator.sclite_utils import (
    TRNFormat,
    sclite_trn,
    parse_sclite_summary,
)
from sj_utils.file.yaml import read_yaml_namespace, read_yaml, YamlSaver
from sj_utils.logger import generate
from sj_utils.evaluator import TimeChecker
from sj_utils.collection import SafetyDict
from sj_utils.typing import override
from sj_utils.audio import load_audio_from_mp4

from rt_whisper_optimizer.data import TEMPLATE, StudyParam
from rt_whisper_optimizer.service import (
    get_token_saver_loader_transcriber,
    get_rt_whisper_transcriber,
    normalize_text,
)

if TYPE_CHECKING:
    pass


class Optimizer(ABC):
    def __init__(
        self,
        data_path: Path,
        study_path: Path,
        output_path: Path,
        cache_storage: Path | None = None,
    ):
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        if data_path.is_file():
            raise ValueError(
                f"Data path {data_path} should be a directory, not a file."
            )
        if study_path.exists() and study_path.is_file():
            raise ValueError(
                f"Study path {study_path} should be a directory, not a file."
            )

        config = read_yaml_namespace(Path(os.getenv("CONFIG_PATH", "config.yml")))
        logger = generate(
            "optimizer",
            level=config.rt_whisper_optimizer.log.level,
            path=Path(config.rt_whisper_optimizer.log.dir_path),
            file_log_level=config.rt_whisper_optimizer.log.file_level,
        )

        cache_storage.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        self.data_path = data_path
        self.study_path = study_path
        self.output_path = output_path
        self.cache_storage = cache_storage
        self.logger = logger

    @staticmethod
    def get_example() -> dict:
        return copy.deepcopy(TEMPLATE)

    def optimize(self, instructions: Path | dict) -> None:
        if isinstance(instructions, Path):
            instructions = read_yaml(instructions)

        description = instructions["description"]
        optimizer = instructions["optimizer"]
        sample_rate = optimizer["model_sample_rate"]
        batch_size = optimizer["batch_size"]
        max_study_steps = optimizer["max_study_steps"]
        study_file_name = optimizer["study_file_name"]
        backup = optimizer["backup"]
        use_cache = optimizer["use_cache"]
        use_prompt = optimizer["use_prompt"]
        random_seed = optimizer["random_seed"]
        study_param = instructions["study"]

        study_param = StudyParam(study_param)
        yaml_saver = YamlSaver(description)
        rng = np.random.default_rng(random_seed)
        save_study_path = self.study_path / f"{study_file_name}.pkl"
        if save_study_path.exists():
            self.logger.warning(
                f"Study path {save_study_path} already exists. Overwriting."
            )

        transcriber = self.__get_transcriber(sample_rate, rng, use_prompt, use_cache)
        objective = self._get_objective_function(study_param, transcriber, batch_size)

        if save_study_path.exists():
            study = joblib.load(save_study_path)
        else:
            study = optuna.create_study(direction="minimize")

        while len(study.trials) < max_study_steps:
            if backup:
                joblib.dump(
                    study, save_study_path.parent / f"{save_study_path.stem}.bak.pkl"
                )
            study.optimize(objective, n_trials=5)
            joblib.dump(study, save_study_path)

        max10study = optuna.create_study(direction=study.direction)
        max10study.add_trials([t for t in study.trials if t.values[0] <= 10])
        max15study = optuna.create_study(direction=study.direction)
        max15study.add_trials([t for t in study.trials if t.values[0] <= 15])
        max20study = optuna.create_study(direction=study.direction)
        max20study.add_trials([t for t in study.trials if t.values[0] <= 20])

        self.__save_plot(study)
        self.__save_plot(max10study, "max10")
        self.__save_plot(max15study, "max15")
        self.__save_plot(max20study, "max20")

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        top_trials = sorted(completed_trials, key=lambda t: t.values[0])[:30]
        cnt_time = time.strftime("%Y%m%d_%H%M%S")
        for trial in top_trials:
            suggested_params = study_param.set_suggest(trial.params)
            yaml_saver.save(
                suggested_params,
                self.output_path
                / f"trial_wer{str(trial.values[0]).replace('.', 'o')}_{trial.number}_{cnt_time}.yaml",
            )

    def __save_plot(self, study: optuna.Study, extra_name: str = "") -> None:
        if extra_name:
            extra_name = f"_{extra_name}"

        cnt_time = time.strftime("%Y%m%d_%H%M%S")

        pi = self.output_path / f"param_importances{extra_name}_{cnt_time}.html"
        ps = self.output_path / f"param_slices{extra_name}_{cnt_time}.html"
        pc = self.output_path / f"parallel_coordinate{extra_name}_{cnt_time}.html"

        fig_pi = vis.plot_param_importances(study)
        fig_pi.write_html(pi)
        fig_ps = vis.plot_slice(study)
        fig_ps.write_html(ps)
        fig_ppc = vis.plot_parallel_coordinate(study)
        fig_ppc.write_html(pc)

    def __get_transcriber(
        self,
        sample_rate: int,
        rng: np.random.Generator,
        use_prompt: bool,
        use_cache: bool,
    ):
        if use_cache:
            if use_prompt:
                self.logger.warning(
                    "Using cache with prompt is not supported. Using default transcriber."
                )
            return self._get_save_loader(sample_rate, rng)
        return self._get_streamer(sample_rate, use_prompt, rng)

    @abstractmethod
    def _get_save_loader(
        self, sr: int, rng: np.random.Generator | np.random.RandomState = np.random
    ) -> Callable[[Path, TimeChecker], str]: ...

    @abstractmethod
    def _get_streamer(
        self,
        sr: int,
        use_prompt: bool,
        rng: np.random.Generator | np.random.RandomState = np.random,
    ) -> Callable[[Path, TimeChecker], str]: ...

    @abstractmethod
    def _get_objective_function(
        self,
        study: StudyParam,
        transcriber: Callable[[Path, TimeChecker], str],
        batch_size: int,
    ) -> Callable[[optuna.Trial], float]: ...


class ESICOptimizer(Optimizer):

    @lru_cache(maxsize=1024)
    def __load_audio(self, src: Path, sr: int) -> tuple[np.ndarray, int]:
        return load_audio_from_mp4(src, sr)

    @override
    def _get_save_loader(
        self, sr: int, rng: np.random.Generator | np.random.RandomState = np.random
    ) -> Callable[[Path, TimeChecker], str]:
        return get_token_saver_loader_transcriber(
            self.data_path,
            self.cache_storage,
            sr,
            self.__load_audio,
            rng=rng,
        )

    @override
    def _get_streamer(
        self,
        sr: int,
        use_prompt: bool,
        rng: np.random.Generator | np.random.RandomState = np.random,
    ) -> Callable[[Path, TimeChecker], str]:
        return get_rt_whisper_transcriber(
            sr,
            self.__load_audio,
            use_prompt=use_prompt,
            rng=rng,
        )

    @override
    def _get_objective_function(
        self,
        study: StudyParam,
        transcriber: Callable[[Path, TimeChecker], str],
        batch_size: int,
    ) -> Callable[[optuna.Trial], float]:
        from sj_ai_utils.datasets.esic_v1 import search_all_data, search_file_from_dir

        data_folders = search_all_data(self.data_path)

        def objective(trial: optuna.Trial) -> float:
            hyperparameter = SafetyDict(study.suggest_all(trial))

            samples = random.sample(data_folders, min(len(data_folders), batch_size))

            data = {}
            for sample in samples:
                key = sample.parent.stem + "_" + sample.stem
                trans_txt = search_file_from_dir(sample, "o")
                txt = trans_txt.read_text(encoding="utf-8")
                txt = normalize_text(txt)
                ref = TRNFormat(id=key, text=txt)

                pred_txt = transcriber(
                    search_file_from_dir(sample, "mp4"),
                    TimeChecker(),
                    hyperparameter,
                    int(hyperparameter["asr"]["max_overlap_duration"]),
                )
                pred_txt = normalize_text(pred_txt)
                hyp = TRNFormat(id=key, text=pred_txt)

                data[key] = {"ref": ref, "hyp": hyp}

            concat_result = {}
            for value in data.values():
                for k, v in value.items():
                    if k not in concat_result:
                        concat_result[k] = []
                    concat_result[k].append(v)

            output = sclite_trn(concat_result["ref"], concat_result["hyp"])
            result = parse_sclite_summary(output)

            return result["wer_percent"]

        return objective


class LibriOptimizer(Optimizer):

    @lru_cache(maxsize=1024)
    def __load_audio(self, src: Path, sr: int) -> tuple[np.ndarray, int]:
        return librosa.load(src, sr=sr)

    @override
    def _get_save_loader(
        self, sr: int, rng: np.random.Generator | np.random.RandomState = np.random
    ) -> Callable[[Path, TimeChecker], str]:
        return get_token_saver_loader_transcriber(
            self.data_path,
            self.cache_storage,
            sr,
            self.__load_audio,
            rng=rng,
        )

    @override
    def _get_streamer(
        self,
        sr: int,
        use_prompt: bool,
        rng: np.random.Generator | np.random.RandomState = np.random,
    ) -> Callable[[Path, TimeChecker], str]:
        return get_rt_whisper_transcriber(
            sr,
            self.__load_audio,
            use_prompt=use_prompt,
            rng=rng,
        )

    @override
    def _get_objective_function(
        self,
        study: StudyParam,
        transcriber: Callable[[Path, TimeChecker], str],
        batch_size: int,
    ) -> Callable[[optuna.Trial], float]:
        from sj_ai_utils.datasets.libri_speech_asr_corpus import (
            search_all_data,
            trans_txt_to_sclite_trn,
        )

        data_folders = search_all_data(self.data_path)

        def objective(trial: optuna.Trial) -> float:
            hyperparameter = SafetyDict(study.suggest_all(trial))

            samples = random.sample(data_folders, min(len(data_folders), batch_size))

            data = {}
            for sample in samples:
                trans_txt = next(sample.glob("*.trans.txt"))
                ref = trans_txt_to_sclite_trn(trans_txt, normalize_text)
                hyp = [
                    TRNFormat(
                        id=flac.stem,
                        text=normalize_text(
                            transcriber(
                                flac,
                                TimeChecker(),
                                hyperparameter,
                                int(hyperparameter["asr"]["max_overlap_duration"]),
                            )
                        ),
                    )
                    for flac in sorted(sample.glob("*.flac"))
                ]
                data[sample.stem] = {"ref": ref, "hyp": hyp}

            concat_result = {}
            for value in data.values():
                for k, v in value.items():
                    if k not in concat_result:
                        concat_result[k] = []
                    concat_result[k].extend(v)

            output = sclite_trn(concat_result["ref"], concat_result["hyp"])
            result = parse_sclite_summary(output)

            return result["wer_percent"]

        return objective


__all__ = [
    "Optimizer",
    "ESICOptimizer",
    "LibriOptimizer",
]
