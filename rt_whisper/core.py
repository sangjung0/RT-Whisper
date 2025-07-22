import os
from pathlib import Path

from sj_utils.collection import SafetyDict
from sj_utils.file.yaml import read_yaml_namespace, read_yaml
from sj_utils.logger import generate

from rt_whisper import RTWhisperLogger


config = read_yaml_namespace(Path(os.getenv("CONFIG_PATH", "config.yml")))
hyperparameter = SafetyDict(
    read_yaml(Path(config.rt_whisper.default_hyperparameter_path))
)
logger = RTWhisperLogger(
    generate(
        "core",
        level=config.rt_whisper.log.level,
        path=Path(config.rt_whisper.log.dir_path),
        file_log_level=config.rt_whisper.log.file_level,
    )
)

__all__ = [
    "config",
    "hyperparameter",
    "logger",
]
