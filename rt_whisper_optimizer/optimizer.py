from __future__ import annotations
from typing import TYPE_CHECKING

import os
import copy
import jiwer
import torch

import numpy as np
import nevergrad as ng

from abc import ABC
from pathlib import Path
from typing import Callable

from sj_utils.file.yaml import read_yaml_namespace, read_yaml, YamlSaver
from sj_utils.logger import generate
from sj_utils.evaluator import TimeChecker
from sj_utils.collection import SafetyDict

from sj_ai_utils.datasets import Dataset

from rt_whisper.models import BoundaryWordFilter

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
        study_path: Path,  # 최적화 객체 저장/백업 경로
        output_path: Path,  # 결과 저장 경로
        instructions: Path | dict,
        cache_storage: Path | None = None,  # dataset 모델 결과 캐싱 경로
        step: int = 0,
    ):
        if study_path.exists() and study_path.is_file():
            raise ValueError(
                f"Study path {study_path} should be a directory, not a file."
            )
        if isinstance(instructions, Path):
            instructions = read_yaml(instructions)

        config = read_yaml_namespace(Path(os.getenv("CONFIG_PATH", "config.yml")))
        logger = generate(
            "optimizer",
            level=config.rt_whisper_optimizer.log.level,
            path=Path(config.rt_whisper_optimizer.log.dir_path),
            file_log_level=config.rt_whisper_optimizer.log.file_level,
        )

        output_path.mkdir(parents=True, exist_ok=True)
        cache_storage.mkdir(parents=True, exist_ok=True)

        self.study_path = study_path
        self.output_path = output_path
        self.cache_storage = cache_storage
        self.instructions = instructions
        self.logger = logger
        self.step = step

    @staticmethod
    def get_example() -> dict:
        return copy.deepcopy(TEMPLATE)

    def optimize(self, dataset: Dataset) -> None:
        description = self.instructions["description"]
        optimizer = self.instructions["optimizer"]
        study_param = self.instructions["study"]

        algo = optimizer["algo"].lower()
        sample_rate = optimizer["model_sample_rate"]
        max_study_steps = optimizer["max_study_steps"]
        # study_file_name = optimizer["study_file_name"]
        # backup = optimizer["backup"]
        use_cache = optimizer["use_cache"]
        use_prompt = optimizer["use_prompt"]
        random_seed = optimizer["random_seed"]
        chunk_size = optimizer["chunk_size"]
        language = optimizer["language"]
        sigma = optimizer["sigma"]
        top_k = optimizer["top_k"]

        study_param = StudyParam(study_param)
        yaml_saver = YamlSaver(description)
        rng = np.random.default_rng(random_seed)
        dataset.sample_rate = sample_rate
        # backup_path = self.study_path / study_file_name

        transcriber = self._get_transcriber(
            use_prompt, use_cache, chunk_size, language, rng
        )
        objective = self._get_objective_function(dataset, study_param, transcriber)

        param = (
            ng.p.Array(init=study_param.get_param())
            .set_bounds(-1, 1)
            .set_mutation(sigma=sigma)
        )

        history = []

        def add_history(_, param, loss):
            self.logger.info(f"Step {len(history)+1}: loss={loss}")
            history.append({"loss": loss, "param": param})

        if algo == "cma":
            optimizer = ng.optimizers.CMA(parametrization=param, budget=max_study_steps)
        elif algo == "spsa":
            optimizer = ng.optimizers.SPSA(
                parametrization=param, budget=max_study_steps
            )
        else:
            raise ValueError(f"Unsupported optimization algorithm: {algo}")
        # optimizer.enable_pickling()
        optimizer.register_callback("tell", add_history)

        optimizer.minimize(objective)

        extracted = set()
        history.sort(key=lambda x: x["loss"])
        for i, h in enumerate(history[:top_k]):
            recommended_param = study_param.set_param(h["param"].args[0])
            key = study_param.get_study_key()
            if key in extracted:
                continue
            file_name = f"{i+1:03}_{str(round(h['loss'], 3)).replace('.', '_')}"

            for key in study_param.train_model_keys:
                model_file_name = f"{file_name}_{key}.pth"
                study_param.model_objs[key].save(self.output_path / model_file_name)

            if len(study_param.train_study_keys) > 0:
                yaml_saver.save(
                    recommended_param,
                    self.output_path / f"{file_name}.yaml",
                )
            extracted.add(key)

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
            head_model: BoundaryWordFilter = None,
            tail_model: BoundaryWordFilter = None,
        ):
            return t(
                audio,
                transcribe_time,
                hyperparameter=hyperparameter,
                head_model=head_model,
                tail_model=tail_model,
            )

        return transcriber

    def _get_objective_function(
        self,
        dataset: Dataset,
        study: StudyParam,
        transcriber: Callable[[np.ndarray, Path, TimeChecker, SafetyDict, int], str],
    ) -> Callable[[np.ndarray], float]:

        head_model = BoundaryWordFilter()
        tail_model = BoundaryWordFilter()
        cache = {}

        def objective(param: np.ndarray) -> float:
            hyperparameter = SafetyDict(study.set_param(param))
            key = study.get_study_key()
            if key in cache:
                return cache[key]

            overlap = int(hyperparameter["asr"]["max_overlap_duration"])
            study.model_objs["head_model"].set(head_model)
            study.model_objs["tail_model"].set(tail_model)

            def t(audio: np.ndarray, audio_key: str):
                return transcriber(
                    audio,
                    audio_key,
                    TimeChecker(),
                    overlap=overlap,
                    hyperparameter=hyperparameter,
                    head_model=head_model,
                    tail_model=tail_model,
                )

            refs, hyps = [], []
            for _id, audio, y in dataset:
                txt = normalize_text(y)
                refs.append(txt)

                pred_txt = t(audio=audio, audio_key=_id)
                pred_txt = normalize_text(pred_txt)
                hyps.append(pred_txt)

            out = jiwer.process_words(
                refs,
                hyps,
                reference_transform=jiwer.wer_default,
                hypothesis_transform=jiwer.wer_default,
            )
            errors = out.substitutions + out.deletions + out.insertions
            # wer = jiwer.wer(refs, hyps)
            # cache[key] = wer
            return errors

        return objective


__all__ = [
    "Optimizer",
]
