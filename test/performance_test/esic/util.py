import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/test/performance_test",
]
for path in paths:
    sys.path.append(os.path.abspath(path))

from pathlib import Path
from typing import Any

from sj_ai_utils.datasets.esic_v1.sclite import generate_ref_and_hyp
from sj_utils.collection import SafetyDict

from common_util import (
    whisper_streaming as ws,
    rt_whisper as rw,
    whisper as w,
    evaluate as ev,
)


def whisper(
    data_paths: Any,
    model_size: str = "large-v3",
    language: str = "en",
    test_all: bool = True,
    max_count: int = -1,
):
    return w(
        data_paths=data_paths,
        generate_ref_and_hyp=generate_ref_and_hyp,
        model_size=model_size,
        language=language,
        test_all=test_all,
        max_count=max_count,
    )


def rt_whisper(
    storage: Path,
    data_paths: Any,
    seed: int = 42,
    use_save_loader: bool = True,
    use_prompt: bool = False,
    language: str = "en",
    hyperparameter: Path | SafetyDict = None,
    chunk_size: int = 48000,
    test_all: bool = True,
    max_count: int = -1,
):
    return rw(
        storage=storage,
        data_paths=data_paths,
        generate_ref_and_hyp=generate_ref_and_hyp,
        seed=seed,
        use_save_loader=use_save_loader,
        use_prompt=use_prompt,
        language=language,
        hyperparameter=hyperparameter,
        chunk_size=chunk_size,
        test_all=test_all,
        max_count=max_count,
    )


def whisper_streaming(
    data_paths: Any,
    model_size: str = "large-v3",
    seed: int = 42,
    language: str = "en",
    chunk_size: int = 48000,
    test_all: bool = True,
    max_count: int = -1,
):
    return ws(
        data_paths=data_paths,
        generate_ref_and_hyp=generate_ref_and_hyp,
        model_size=model_size,
        seed=seed,
        language=language,
        chunk_size=chunk_size,
        test_all=test_all,
        max_count=max_count,
    )


def evaluate(
    storage: Path,
    output_path: Path,
    description: str,
    data_paths: Any,
    models: list[str] = ["rt_whisper"],
    model_size: str = "large-v3",
    language: str = "en",
    test_all: bool = True,
    max_count: int = -1,
    seed: int = 42,
    use_save_loader: bool = True,
    use_prompt: bool = False,
    hyperparameter: Path = None,
    chunk_size: int = 48_000,
):
    return ev(
        storage=storage,
        output_path=output_path,
        description=description,
        data_paths=data_paths,
        generate_ref_and_hyp=generate_ref_and_hyp,
        models=models,
        model_size=model_size,
        language=language,
        test_all=test_all,
        max_count=max_count,
        seed=seed,
        use_save_loader=use_save_loader,
        use_prompt=use_prompt,
        hyperparameter=hyperparameter,
        chunk_size=chunk_size,
    )


__all__ = ["evaluate", "whisper_streaming", "rt_whisper", "whisper"]
