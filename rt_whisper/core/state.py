import os
from pathlib import Path

from rt_whisper.util import ReadYaml, SafetyDict


config = ReadYaml(Path(os.getenv("CONFIG_PATH", "config.yml"))).namespace
hyperparameter = SafetyDict(
    data={
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
        "max_overlap_duration": 96000,
    }
)
