from __future__ import annotations
from typing import TYPE_CHECKING

import os
import copy
import jiwer

import numpy as np
import nevergrad as ng

from abc import ABC
from pathlib import Path
from typing import Callable

from sj_utils.file.yaml import read_yaml_namespace, YamlSaver
from sj_utils.logger import generate
from sj_utils.collection import SafetyDict

from sj_ai_utils.datasets import Dataset

from rt_whisper.models.boundary_word_filter import BoundaryWordFilter

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
        instructions: dict,
        cache_storage: Path | None = None,  # dataset 모델 결과 캐싱 경로
        step: int = 0,
    ):
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

    def optimize(
        self, dataset: Dataset, plot_history: bool = True, log_step: int = 20
    ) -> None:
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
        best_loss = {"loss": float("inf"), "param": None}

        def add_history(_, param, loss):
            self.logger.debug(
                f"✅ Step {len(history)+1}: loss={loss} | best_loss={best_loss['loss']}"
            )
            if loss < best_loss["loss"]:
                best_loss["loss"] = loss
                best_loss["param"] = param
            if len(history) % log_step == 0:
                self.logger.info(
                    f"✅ Step {len(history)+1}: loss={loss} | best_loss={best_loss['loss']}"
                )
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

        self.logger.info(
            f"🟢 Starting optimization with algorithm={algo}, max_study_steps={max_study_steps}, sample_rate={sample_rate}, use_cache={use_cache}, use_prompt={use_prompt}, random_seed={random_seed}, chunk_size={chunk_size}, language={language}, sigma={sigma}, top_k={top_k}"
        )
        optimizer.minimize(objective)

        extracted = set()
        for i, h in enumerate(sorted(history, key=lambda x: x["loss"])[:top_k]):
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

        if plot_history:
            self.__plot_history(history)

        return history

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
            overlap: int = None,
            hyperparameter: SafetyDict = None,
            model: BoundaryWordFilter = None,
        ):
            return t(
                audio,
                hyperparameter=hyperparameter,
                model=model,
            )

        return transcriber

    def _get_objective_function(
        self,
        dataset: Dataset,
        study: StudyParam,
        transcriber: Callable[[np.ndarray, Path, int, SafetyDict, BoundaryWordFilter], str],
    ) -> Callable[[np.ndarray], float]:

        model = BoundaryWordFilter()
        # cache = {}

        def objective(param: np.ndarray) -> float:
            hyperparameter = SafetyDict(study.set_param(param))
            # key = study.get_study_key()
            # if key in cache:
            # return cache[key]

            overlap = int(hyperparameter["asr"]["max_overlap_duration"])
            study.model_objs["model"].set(model)

            def t(audio: np.ndarray, audio_key: str):
                return transcriber(
                    audio,
                    audio_key,
                    overlap=overlap,
                    hyperparameter=hyperparameter,
                    model=model,
                )

            refs, hyps = [], []
            for _id, audio, y in dataset:
                txt = normalize_text(y)
                refs.append(txt)

                pred_txt = t(audio=audio, audio_key=_id)
                pred_txt = normalize_text(pred_txt)
                hyps.append(pred_txt)

            wer = jiwer.wer(refs, hyps)
            # cache[key] = wer
            return wer

        return objective

    def __plot_history(self, history: list[dict[str, float | np.ndarray]]):
        import matplotlib.pyplot as plt

        losses = [h["loss"] for h in history]
        steps = range(1, len(losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, color="blue", linewidth=2, zorder=1)
        plt.scatter(steps, losses, color="red", s=10, zorder=2)

        plt.title("Optimization History")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()


__all__ = [
    "Optimizer",
]
