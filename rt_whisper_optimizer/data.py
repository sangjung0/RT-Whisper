from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sj_utils.wrapper import make_float_like, make_int_like
from sj_utils.typing import override

if TYPE_CHECKING:
    from optuna import Trial

TEMPLATE = {
    "description": "Test",
    "optimizer": {
        "model_sample_rate": 16000,
        "batch_size": 16,
        "max_study_steps": 3000,
        "study_file_name": "study.pky",
        "backup": True,
        "use_cache": True,
        "use_prompt": True,
        "random_seed": 42,
    },
    "study": {
        "whisper": {
            "model_options": {
                "model_size_or_path": "large-v3",
                "device": "cuda",
                "compute_type": "float16",
            },
            "transcribe_options": {
                "beam_size": 5,
                "vad_filter": False,
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            },
        },
        "silero_vad": {"model_options": {}, "run_options": {}},
        "asr": {
            "max_overlap_duration": {
                "is_train": True,
                "key": "overlap_duration",
                "value": 96000,
                "minimum": 0,
                "maximum": 112000,
                "step": 16000,
            }
        },
        "position_weighted_filter": {
            "boundary": {
                "is_train": True,
                "key": "boundary",
                "value": 8000,
                "minimum": 0,
                "maximum": 19520,
                "step": 160,
            }
        },
        "duration_filter": {
            "z_thresh": {
                "default": 2.0,
                "en": {
                    "is_train": True,
                    "key": "df_z_thresh",
                    "value": 3.0,
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "step": 0.01,
                },
            },
            "min_dur": {
                "default": 160,
                "en": {
                    "is_train": True,
                    "key": "df_min_dur",
                    "value": 160,
                    "minimum": 0,
                    "maximum": 1920,
                    "step": 160,
                },
            },
        },
        "probability_filter": {
            "z_thresh": {
                "default": 3.0,
                "ko": {
                    "is_train": True,
                    "key": "pf_z_thresh",
                    "value": 3.0,
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "step": 0.01,
                },
            },
            "min_prob": {
                "default": 1.0,
                "en": {
                    "is_train": True,
                    "key": "pf_min_prob",
                    "value": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "step": 0.01,
                },
            },
        },
        "selector": {
            "iou_threshold": {
                "default": 0.5,
                "en": {
                    "is_train": True,
                    "key": "iou_threshold",
                    "value": 0.5,
                    "minimum": 0.05,
                    "maximum": 0.4,
                    "step": 0.01,
                },
            },
            "cos_threshold": {
                "default": 0.5,
                "en": {
                    "is_train": True,
                    "key": "cos_threshold",
                    "value": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "step": 0.01,
                },
            },
            "padding": {
                "default": 3200,
                "en": {
                    "is_train": True,
                    "key": "padding",
                    "value": 1600,
                    "minimum": 0,
                    "maximum": 1760,
                    "step": 160,
                },
            },
        },
    },
}


@dataclass(slots=True)
class Param(ABC):
    key: str
    minimum: float | int
    maximum: float | int
    step: float | int
    is_train: bool = field(default=True)
    value: float | int | None = field(default=None)

    @classmethod
    def from_dict(cls, data: dict) -> Param:
        return cls(
            key=data["key"],
            minimum=data["minimum"],
            maximum=data["maximum"],
            step=data["step"],
            is_train=data.get("is_train", True),
            value=data.get("value", None),
        )

    @abstractmethod
    def suggest(self, trial: Trial) -> float | int: ...


@make_float_like
@dataclass(slots=True)
class FloatParam(Param):
    @override
    def suggest(self, trial: Trial) -> float:
        if self.is_train:
            self.value = trial.suggest_float(
                self.key, self.minimum, self.maximum, step=self.step
            )
        return self.value


@make_int_like
@dataclass(slots=True)
class IntParam(Param):
    @override
    def suggest(self, trial: Trial) -> int:
        if self.is_train:
            self.value = trial.suggest_int(
                self.key, self.minimum, self.maximum, step=self.step
            )
        return self.value


@dataclass(slots=True)
class StudyParam:
    original_study: dict
    study_objs: dict = field(default_factory=dict, init=False)
    suggested_params: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        if not isinstance(self.original_study, dict):
            raise TypeError("original_study must be a dictionary.")
        self.suggested_params = self.__find_study_obj(self.original_study)

    def __is_study_param(self, value: dict) -> Param | None:
        if "is_train" not in value:
            return None
        key, step = value["key"], value["step"]

        if key in self.study_objs:
            raise ValueError(f"Duplicate key '{key}' found in study parameters.")
        elif isinstance(step, int):
            param = IntParam.from_dict(value)
        else:
            param = FloatParam.from_dict(value)
        self.study_objs[param.key] = param
        return param

    def __find_study_obj(self, obj: dict) -> dict:
        copy_dict = {}
        for key, value in obj.items():
            if isinstance(value, dict):
                param = self.__is_study_param(value)
                value = param or self.__find_study_obj(value)
            elif isinstance(value, list):
                value = [
                    self.__find_study_obj(item) if isinstance(item, dict) else item
                    for item in value
                ]
            copy_dict[key] = value
        return copy_dict

    def suggest_all(self, trial: Trial) -> dict:
        for param in self.study_objs.values():
            param.suggest(trial)
        return self.suggested_params

    def set_suggest(self, trial_params: dict) -> dict:
        for key, value in trial_params.items():
            self.study_objs[key].value = value
        return self.suggested_params


__all__ = [
    "TEMPLATE",
    "Param",
    "FloatParam",
    "IntParam",
    "StudyParam",
]
