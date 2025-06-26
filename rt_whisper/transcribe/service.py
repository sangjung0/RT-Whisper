from pathlib import Path

from sj_utils.collection_utils import SafetyDict
from sj_utils.file import ReadYaml

from rt_whisper.core.state import hyperparameter as __default_hyperparameter


def get_hyperparameter(
    hyperparameter_path: Path | str | None = None,
):
    if hyperparameter_path is None:
        return __default_hyperparameter
    else:
        if isinstance(hyperparameter_path, str):
            hyperparameter_path = Path(hyperparameter_path)
        return SafetyDict(ReadYaml(hyperparameter_path).dict)
