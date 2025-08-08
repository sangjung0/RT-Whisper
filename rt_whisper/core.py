import os
from pathlib import Path

from sj_utils.reference import get_top_package_root
from sj_utils.collection import SafetyDict
from sj_utils.file.yaml import read_yaml_namespace, read_yaml
from sj_utils.logger import generate

from rt_whisper.logger import RTWhisperLogger

package_path = get_top_package_root()


def __core():
    config_path = Path(os.getenv("CONFIG_PATH", "config.yml"))
    if not config_path.exists():
        config_path = package_path.parent / "config.yml"

    config = read_yaml_namespace(config_path)
    if getattr(config, "rt_whisper", None) is None:
        config_path = package_path.parent / "config.yml"
        config = read_yaml_namespace(config_path)
        if getattr(config, "rt_whisper", None) is None:
            raise ValueError(
                "RTWhisper configuration not found in the provided config file."
            )

    hyperparameter_path = Path(config.rt_whisper.default_hyperparameter_path)
    if not hyperparameter_path.exists():
        hyperparameter_path = package_path.parent / "default_hyperparameter.yml"
        if not hyperparameter_path.exists():
            raise FileNotFoundError(
                f"Default hyperparameter file not found at {hyperparameter_path}"
            )

    hyperparameter = SafetyDict(read_yaml(hyperparameter_path))

    logger = RTWhisperLogger(
        generate(
            "RTWhisper",
            level=config.rt_whisper.log.level,
            path=Path(config.rt_whisper.log.dir_path),
            file_log_level=config.rt_whisper.log.file_level,
        )
    )

    return config, hyperparameter, logger


config, hyperparameter, logger = __core()

__all__ = [
    "config",
    "hyperparameter",
    "logger",
]
