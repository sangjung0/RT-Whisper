import os
from pathlib import Path

from sj_utils.collection_utils import SafetyDict
from sj_utils.file import ReadYaml


config = ReadYaml(Path(os.getenv("CONFIG_PATH", "config.yml"))).namespace
hyperparameter = SafetyDict(ReadYaml(Path(config.rt_whisper.default_hyperparameter_path)).dict)
