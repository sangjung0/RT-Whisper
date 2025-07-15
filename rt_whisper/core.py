import os
from pathlib import Path

from sj_utils.collection_utils import SafetyDict
from sj_utils.file import ReadYaml
from sj_utils.logger import generate


config = ReadYaml(Path(os.getenv("CONFIG_PATH", "config.yml"))).namespace
hyperparameter = SafetyDict(
    ReadYaml(Path(config.rt_whisper.default_hyperparameter_path)).dict
)
logger = generate(
    "core",
    level=config.rt_whisper.log.level,
    path=Path(config.rt_whisper.log.dir_path),
    file_log_level=config.rt_whisper.log.file_level,
)

__all__ = [
    "config",
    "hyperparameter",
    "logger",
]
