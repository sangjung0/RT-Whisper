from logging import Logger
from typing import Any

from .BaseLogger import BaseLogger


HYPERPARAMETERS = {
    "sentence_max_prev_sentence": 1,
    "weighted_and_offset_token_boundary": 8000,
    "duration_filter_z": {
        "default": 2.0,
        "ko": 2.0,
        "en": 2.0,
    },
    "probability_filter": {
        "z": {
            "default": 2.0,
            "ko": 2.0,
            "en": 2.0,
        },
        "min_prob": {
            "default": 1.0,
            "ko": 0.4,
            "en": 0.4,
        },
    },
    "selector": {
        "search_range_sc": {
            "default": 24000,
            "ko": 24000,
            "en": 24000,
        },
        "threshold": {
            "default": 0.5,
            "ko": 0.25,
            "en": 0.5,
        },
        "padding": {
            "default": 3200,
            "ko": 3200,
            "en": 3200,
        },
        "tolerance": {
            "default": 8000,
            "ko": 8000,
            "en": 8000,
        },
    },
    "classifier_max_prev_sc": {"default": 96000},
}


class Hyperparameters(dict):
    @BaseLogger.object
    def __init__(
        self,
        default_value: Any = None,
        setting: dict = HYPERPARAMETERS,
        base_logger: Logger = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__DEFAULT_VALUE = default_value
        self.__logger = base_logger
        if setting:
            self.update(self.__upload(setting))

    def __getitem__(self, key):
        if key not in self:
            self.__logger.warning(
                f"Key '{key}' not found in SafetyDict. Returning default value."
            )
        return super().get(key, self.__DEFAULT_VALUE)

    def __setitem__(self, key, value):
        raise NotImplementedError("SafetyDict is read-only")

    def __upload(self, d: dict):
        default = d["default"] if "default" in d else None
        bucket = Hyperparameters(default, setting=None)
        for key, value in d.items():
            if isinstance(value, dict):
                super(Hyperparameters, bucket).__setitem__(key, self.__upload(value))
            else:
                super(Hyperparameters, bucket).__setitem__(key, value)
        return bucket
