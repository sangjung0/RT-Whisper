from pathlib import Path

from sj_utils.collection_utils import SafetyDict
from sj_utils.file import ReadYaml

from rt_whisper.core.state import hyperparameter as default_hyperparameter


def init_hyperparameter(
    hyperparameter: SafetyDict | Path | str | None = None,
):
    if hyperparameter is None:
        hyperparameter = default_hyperparameter
    elif isinstance(hyperparameter, str) or isinstance(hyperparameter, Path):
        hyperparameter = SafetyDict(ReadYaml(Path(hyperparameter)).dict)
    elif not isinstance(hyperparameter, SafetyDict):
        raise TypeError(
            f"Expected hyperparameter to be SafetyDict, Path, or str, got {type(hyperparameter)}"
        )

    return hyperparameter


__all__ = [
    "init_hyperparameter",
]
