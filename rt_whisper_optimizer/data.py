from __future__ import annotations
from typing import TYPE_CHECKING

import yaml
import numpy as np
import torch

from torch import nn
from pathlib import Path
from abc import ABC
from dataclasses import dataclass, field
from typing_extensions import override
from typing import ClassVar

from sj_utils.wrapper import make_float_like, make_int_like

if TYPE_CHECKING:
    pass

TEMPLATE = {
    "description": "Test",
    "optimizer": {
        "model_sample_rate": 16000,
        "max_study_steps": 3000,
        # "study_file_name": "study.pky",
        # "backup": True,
        "use_cache": True,
        "use_prompt": True,
        "random_seed": 42,
        "chunk_size": 48000,
        "language": "en",
        "sigma": 0.1,
        "top_k": 10,
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
            "max_prompt_word": 10,
            "max_overlap_duration": {
                "is_train": True,
                "key": "overlap_duration",
                "value": 96000,
                "minimum": 0,
                "maximum": 112000,
                "step": 16000,
            },
        },
        "position_weighted_filter": {
            "head_model": {
                "key": "head_model",
                "is_train": True,
                "model_path": "path/to/head_model/params",
                "device": "cpu",
                "maximum": 1.0,
                "minimum": 0.0,
            },
            "tail_model": {
                "key": "tail_model",
                "is_train": True,
                "model_path": "path/to/tail_model/params",
                "device": "cpu",
                "maximum": 1.0,
                "minimum": 0.0,
            },
            "head_boundary": {
                "is_train": True,
                "key": "head_boundary",
                "value": 8000,
                "minimum": 0,
                "maximum": 19520,
                "step": 160,
            },
            "tail_boundary": {
                "is_train": True,
                "key": "tail_boundary",
                "value": 8000,
                "minimum": 0,
                "maximum": 19520,
                "step": 160,
            },
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
                "en": {
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
            "m": {
                "is_train": True,
                "key": "m",
                "value": 0.5,
                "minimum": 0,
                "maximum": 1,
                "step": 0.001,
            },
            "p": {
                "is_train": True,
                "key": "p",
                "value": 0.5,
                "minimum": 0,
                "maximum": 0,
                "step": 0.001,
            },
            "c": {
                "is_train": True,
                "key": "c",
                "value": 0.5,
                "minimum": 0,
                "maximum": 1,
                "step": 0.001,
            },
            "s": {
                "is_train": True,
                "key": "s",
                "value": 0.5,
                "minimum": 0,
                "maximum": 0,
                "step": 0.001,
            },
            "i": {
                "is_train": True,
                "key": "i",
                "value": 0.5,
                "minimum": 0,
                "maximum": 1,
                "step": 0.001,
            },
            "algo": "op",
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
            "token_group_size": 2,
        },
    },
}


@dataclass(slots=True)
class Param(ABC):
    key: str
    minimum: float | int
    maximum: float | int
    step: float | int = field(default=0)
    is_train: bool = field(default=True)
    value: float | int | None = field(default=None)

    @override
    def __post_init__(self):
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be less than maximum.")
        if self.value is None:
            self.value = self.minimum
        if not (self.minimum <= self.value <= self.maximum):
            raise ValueError("value must be between minimum and maximum.")
        if self.step < 0:
            raise ValueError("step must be non-negative.")

    @classmethod
    def from_dict(cls, data: dict) -> Param:
        return cls(
            key=data["key"],
            minimum=data["minimum"],
            maximum=data["maximum"],
            step=data.get("step", 0),
            is_train=data.get("is_train", True),
            value=data.get("value", None),
        )

    def get_value(self) -> float | int:
        return (self.value - self.minimum) / (self.maximum - self.minimum) * 2 - 1

    def set_value(self, value: float | int) -> None:
        self.value = (value + 1) / 2 * (self.maximum - self.minimum) + self.minimum
        if self.step > 0:
            self.value = self.value - (self.value % self.step)


@make_float_like
@dataclass(slots=True)
class FloatParam(Param):
    pass


yaml.add_representer(
    FloatParam, lambda dumper, data: dumper.represent_data(float(data.value))
)


@make_int_like
@dataclass(slots=True)
class IntParam(Param):
    @override
    def set_value(self, value: float | int) -> None:
        super(IntParam, self).set_value(value)
        self.value = int(self.value)


yaml.add_representer(IntParam, lambda dumper, data: dumper.represent_int(data.value))


@dataclass(slots=True)
class ModelParam(Param):
    model_path: Path | str = field(default=None)
    value: np.ndarray | None = field(default=None)
    device: torch.device | str = field(default=torch.device("cpu"))
    model: nn.Module | None = field(default=None, init=False)

    @classmethod
    @override
    def from_dict(cls, data: dict) -> Param:
        return cls(
            key=data["key"],
            minimum=data["minimum"],
            maximum=data["maximum"],
            model_path=data["model_path"],
            step=data.get("step", 0),
            is_train=data.get("is_train", True),
            value=data.get("value", None),
            device=data.get("device", torch.device("cpu")),
        )

    def __str__(self):
        return str(self.model_path)

    @override
    def __post_init__(self):
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        if not (self.model_path.exists() and self.model_path.is_file()):
            raise ValueError(f"Model parameters file {self.model_path} does not exist.")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        # 범용성을 위해 모델 자체 로드
        self.model = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        self.value = torch.cat(
            [p.data.view(-1) for p in self.model.parameters()]
        ).numpy()

    @override
    def get_value(self) -> np.ndarray:
        return super(ModelParam, self).get_value()

    @override
    def set_value(self, value: np.ndarray) -> None:
        return super(ModelParam, self).set_value(value)

    def set(self, model: nn.Module) -> None:
        with torch.no_grad():
            pointer = 0
            for p in model.parameters():
                num_param = p.numel()
                param_values = self.value[pointer : pointer + num_param]
                param_values = param_values.reshape(p.shape)
                p.copy_(torch.from_numpy(param_values).to(p.device))
                pointer += num_param
            if pointer != len(self.value):
                raise ValueError(
                    "The number of parameters in the model does not match the length of the value array."
                )

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Model is not loaded.")
        self.model_path = path
        self.set(self.model)
        torch.save(self.model, path)


yaml.add_representer(
    ModelParam, lambda dumper, data: dumper.represent_data(str(data.model_path))
)


@dataclass(slots=True)
class StudyParam:
    original_study: dict
    study_objs: dict = field(default_factory=dict, init=False)
    model_objs: dict = field(default_factory=dict, init=False)
    suggested_params: dict = field(default_factory=dict, init=False)
    train_study_keys: list[str] = field(default_factory=list, init=False)
    train_model_keys: list[str] = field(default_factory=list, init=False)

    INTEGER: ClassVar[str] = "integer"
    FLOAT: ClassVar[str] = "float"
    MODEL: ClassVar[str] = "model"

    def __post_init__(self):
        if not isinstance(self.original_study, dict):
            raise TypeError("original_study must be a dictionary.")
        self.suggested_params = self.__find_obj(self.original_study)
        self.train_study_keys = [
            k for k in self.study_objs.keys() if self.study_objs[k].is_train
        ]
        self.train_model_keys = [
            k for k in self.model_objs.keys() if self.model_objs[k].is_train
        ]

    def __generate_obj(self, value: dict) -> Param | None:
        if "is_train" not in value:
            return None
        key, dtype = value["key"], value["dtype"]
        if key in self.study_objs or key in self.model_objs:
            raise ValueError(f"Duplicate key '{key}' found in parameters.")
        if dtype not in {self.INTEGER, self.FLOAT, self.MODEL}:
            raise ValueError(
                f"dtype must be one of '{self.INTEGER}', '{self.FLOAT}', or '{self.MODEL}'."
            )

        if dtype == self.INTEGER:
            param = IntParam.from_dict(value)
            self.study_objs[param.key] = param
        elif dtype == self.FLOAT:
            param = FloatParam.from_dict(value)
            self.study_objs[param.key] = param
        else:
            param = ModelParam.from_dict(value)
            self.model_objs[param.key] = param
        return param

    def __find_obj(self, obj: dict) -> dict:
        copy_dict = {}
        for key, value in obj.items():
            if isinstance(value, dict):
                param = self.__generate_obj(value)
                value = param or self.__find_obj(value)
            elif isinstance(value, list):
                value = [
                    self.__find_obj(item) if isinstance(item, dict) else item
                    for item in value
                ]
            copy_dict[key] = value
        return copy_dict

    def get_param(self) -> np.ndarray:
        params = np.asarray(
            [self.study_objs[key].get_value() for key in self.train_study_keys]
        )
        params = np.concatenate(
            [
                params,
                *[self.model_objs[key].get_value() for key in self.train_model_keys],
            ]
        )

        return params

    def set_param(self, param: np.ndarray) -> dict:
        off = 0
        for key in self.train_study_keys:
            self.study_objs[key].set_value(param[off])
            off += 1
        for key in self.train_model_keys:
            self.model_objs[key].set_value(
                param[off : len(self.model_objs[key].value) + off]
            )
            off += len(self.model_objs[key].value)
        if off != len(param):
            raise ValueError("Length of param does not match expected.")
        return self.suggested_params

    def get_study_key(self) -> str:
        # NOTE 이거 쓸 순 있는데, 배열 길어지면 좀 별로일듯
        keys = [str(self.study_objs[key].value) for key in self.train_study_keys] + [
            str(self.model_objs[key].value) for key in self.train_model_keys
        ]
        return "_".join(keys)


__all__ = [
    "TEMPLATE",
    "Param",
    "FloatParam",
    "IntParam",
    "StudyParam",
]
