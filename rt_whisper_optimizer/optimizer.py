from __future__ import annotations
from typing import TYPE_CHECKING

import os
import time
import joblib
import optuna
import copy
import optuna.visualization as vis

import numpy as np

from abc import ABC
from pathlib import Path
from typing import Callable
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from sj_utils.file.yaml import read_yaml_namespace, read_yaml, YamlSaver
from sj_utils.logger import generate
from sj_utils.evaluator import TimeChecker
from sj_utils.collection import SafetyDict

from sj_ai_utils.datasets import Dataset
from sj_ai_utils.evaluator.sclite_utils import (
    TRNFormat,
    sclite_trn,
    parse_sclite_summary,
)

from rt_whisper_optimizer.data import TEMPLATE, StudyParam
from rt_whisper_optimizer.service import (
    get_rt_whisper_transcriber,
    get_token_saver_loader_transcriber,
    normalize_text,
)

if TYPE_CHECKING:
    pass


class Optimizer(ABC):
    def __init__(
        self,
        dataset: Dataset,
        study_path: Path,
        output_path: Path,
        param_cache_path: Path = None,
        cache_storage: Path | None = None,
    ):
        if study_path.exists() and study_path.is_file():
            raise ValueError(
                f"Study path {study_path} should be a directory, not a file."
            )
        if param_cache_path and param_cache_path.exists() and param_cache_path.is_dir():
            raise ValueError(
                f"Parameter cache path {param_cache_path} should be a file, not a directory."
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

        self.datasets = dataset
        self.study_path = study_path
        self.output_path = output_path
        self.param_cache_path = param_cache_path
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
        max_study_steps = optimizer["max_study_steps"]
        study_file_name = optimizer["study_file_name"]
        backup = optimizer["backup"]
        use_cache = optimizer["use_cache"]
        use_prompt = optimizer["use_prompt"]
        random_seed = optimizer["random_seed"]
        chunk_size = optimizer["chunk_size"]
        language = optimizer["language"]
        top_k = optimizer.get("top_k", 30)
        percentiles = optimizer.get("percentiles", [1, 5, 10, 20, 50])
        study_param = instructions["study"]

        study_param = StudyParam(study_param)
        yaml_saver = YamlSaver(description)
        rng = np.random.default_rng(random_seed)
        save_study_path = self.study_path / f"{study_file_name}.pkl"
        backup_path = self.study_path / f"{study_file_name}.bak.pkl"
        if save_study_path.exists():
            self.logger.warning(
                f"Study path {save_study_path} already exists. Overwriting."
            )

        self.datasets.sample_rate = sample_rate

        if self.param_cache_path and self.param_cache_path.exists():
            cache = joblib.load(self.param_cache_path)
        else:
            cache = {}

        transcriber = self._get_transcriber(
            use_prompt, use_cache, chunk_size, language, rng
        )
        objective = self._get_objective_function(
            study_param, transcriber, cache, chunk_size
        )

        if save_study_path.exists() or backup_path.exists():
            try:
                study = joblib.load(save_study_path)
            except Exception as e:
                self.logger.error(f"Failed to load study from {save_study_path}: {e}")
                try:
                    study = joblib.load(backup_path)
                except Exception as e:
                    self.logger.error(f"Failed to load study from {backup_path}: {e}")
                    raise e
        else:
            study = optuna.create_study(
                direction="minimize",
                sampler=TPESampler(
                    multivariate=True, group=True, n_startup_trials=40, seed=random_seed
                ),
                pruner=SuccessiveHalvingPruner(
                    min_resource=1, reduction_factor=3, min_early_stopping_rate=0
                ),
            )

        done = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        remaining = max(0, max_study_steps - done)
        if remaining > 0:

            def _backup_cb(
                study_obj: optuna.Study, trial_obj: optuna.trial.FrozenTrial
            ):
                if trial_obj.number % 10 == 0:
                    if backup:
                        joblib.dump(study_obj, save_study_path)
                    if self.param_cache_path and cache.get(
                        "__optimizer_key_dirty__", False
                    ):
                        del cache["__optimizer_key_dirty__"]
                        joblib.dump(cache, self.param_cache_path)

            study.optimize(objective, n_trials=remaining, callbacks=[_backup_cb])
            joblib.dump(study, save_study_path)
            if self.param_cache_path:
                joblib.dump(cache, self.param_cache_path)

        values = np.array(
            [
                t.values[0]
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        )
        thresholds = [np.percentile(values, p) for p in percentiles]
        percentile_trials = [
            [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.values[0] <= threshold
            ]
            for threshold in thresholds
        ]
        studies = [
            optuna.create_study(direction=study.direction)
            for _ in range(len(percentiles))
        ]
        for s, t in zip(studies, percentile_trials):
            s.add_trials(t)

        self.__save_plot(study)
        for s, p in zip(studies, percentiles):
            self.__save_plot(s, f"p{p}")

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        top_trials = sorted(completed_trials, key=lambda t: t.values[0])[:top_k]
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

    def _get_transcriber(
        self,
        use_prompt: bool,
        use_cache: bool,
        chunk_size: int,
        language: str,
        rng: np.random.Generator,
    ):
        if use_cache:
            if self.cache_storage is None:
                raise ValueError("Cache storage is not set. Cannot use cache.")
            if use_prompt:
                self.logger.warning(
                    "Using cache with prompt is not supported. Using default transcriber."
                )
            return get_token_saver_loader_transcriber(
                self.cache_storage,
                chunk_size_mean=chunk_size,
                rng=rng,
                language=language,
            )

        t = get_rt_whisper_transcriber(
            chunk_size_mean=chunk_size,
            rng=rng,
            language=language,
            use_prompt=use_prompt,
        )

        def transcriber(
            audio: np.ndarray,
            audio_key: Path | str,
            transcribe_time: TimeChecker,
            overlap: int = None,
            hyperparameter: SafetyDict = None,
        ):
            return t(
                audio,
                transcribe_time,
                hyperparameter=hyperparameter,
            )

        return transcriber

    def _get_objective_function(
        self,
        study: StudyParam,
        transcriber: Callable[[np.ndarray, Path, TimeChecker, SafetyDict, int], str],
        cache: dict,
        chunk_size: int,
    ) -> Callable[[optuna.Trial], float]:

        l = len(self.datasets)
        s1_size = int(l * 0.05)
        s2_size = int(l * 0.20)
        s1 = self.datasets[:s1_size]
        s2 = self.datasets[:s2_size]
        s3 = self.datasets

        def _eval(hyperparameter: SafetyDict, dataset: Dataset):
            overlap = int(hyperparameter["asr"]["max_overlap_duration"])
            refs, hyps = [], []
            for _id, audio, y in dataset:
                txt = normalize_text(y)
                ref = TRNFormat(id=_id, text=txt)

                pred_txt = transcriber(
                    audio=audio,
                    audio_key=_id,
                    transcribe_time=TimeChecker(),
                    hyperparameter=hyperparameter,
                    overlap=overlap,
                )
                pred_txt = normalize_text(pred_txt)
                hyp = TRNFormat(id=_id, text=pred_txt)

                refs.append(ref)
                hyps.append(hyp)

            wer = parse_sclite_summary(sclite_trn(refs, hyps))["wer_percent"]
            return wer

        def objective(trial: optuna.Trial) -> float:
            hyperparameter = SafetyDict(study.suggest_all(trial))

            key = self._get_param_key(chunk_size, hyperparameter)
            s1s, s2s, s3s = cache.get(key, (None, None, None))

            s1s = _eval(hyperparameter, s1) if s1s is None else s1s
            trial.report(s1s, step=1)
            if trial.should_prune():
                cache[key] = (s1s, s2s, s3s)
                raise optuna.TrialPruned()

            s2s = _eval(hyperparameter, s2) if s2s is None else s2s
            trial.report(s2s, step=2)
            if trial.should_prune():
                cache[key] = (s1s, s2s, s3s)
                raise optuna.TrialPruned()

            s3s = _eval(hyperparameter, s3) if s3s is None else s3s
            trial.report(s3s, step=3)

            if key not in cache:
                cache[key] = (s1s, s2s, s3s)
                cache["__optimizer_key_dirty__"] = True
            return s3s

        return objective

    def _get_param_key(self, chunk_size: int, hyperparameter: SafetyDict):
        return f"{chunk_size}_{hyperparameter['asr']['max_overlap_duration']*1}_{hyperparameter['position_weighted_filter']['boundary']*1}_{hyperparameter['position_weighted_filter']['exponent']*1}_{hyperparameter['duration_filter']['z_thresh']['en']*1}_{hyperparameter['duration_filter']['min_dur']['en']*1}_{hyperparameter['probability_filter']['z_thresh']['en']*1}_{hyperparameter['probability_filter']['min_prob']['en']*1}_{hyperparameter['selector']['iou_threshold']['en']*1}_{hyperparameter['selector']['cos_threshold']['en']*1}_{hyperparameter['selector']['padding']['en']*1}_{hyperparameter['selector']['token_group_size']*1}"


__all__ = [
    "Optimizer",
]
