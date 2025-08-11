import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/test/performance_test",
    "/workspaces/dev/test/performance_test/libri",
]
for path in paths:
    sys.path.append(os.path.abspath(path))

from pathlib import Path

from sj_utils.collection import SafetyDict

from summarize_util import (
    chunk_test as ct,
    hyperparameter_test as ht,
    generate_statistical_dict,
    generate_statistical_list,
    show_table,
    show_ribbon_plot,
    show_ribbon_plot_with_mm,
    show_line_plot,
)

from util import rt_whisper as rtw, whisper_streaming as ws, whisper as w


def whisper(data_dirs: list[Path], repeat: int = 1, count: int = 1) -> list[dict]:
    result = []
    for _ in range(repeat):
        result.append(w(data_dirs, max_count=count))
    return result


def rt_whisper(
    data_dirs: list[Path],
    hyperparameter: SafetyDict,
    chunk_size: int,
    repeat: int = 1,
    count: int = 1,
    random_seed: int = 42,
) -> list[dict]:
    result = []
    for _ in range(repeat):
        result.append(
            rtw(
                Path("/"),
                Path("/"),
                data_dirs,
                seed=random_seed,
                use_save_loader=False,
                use_prompt=True,
                hyperparameter=hyperparameter,
                chunk_size=chunk_size,
                max_count=count,
            )
        )
    return result


def whisper_streaming(
    data_dirs: list[Path],
    chunk_size: int,
    repeat: int = 1,
    count: int = 1,
    random_seed: int = 42,
) -> list[dict]:
    result = []
    for _ in range(repeat):
        result.append(
            ws(
                data_dirs,
                seed=random_seed,
                chunk_size=chunk_size,
                max_count=count,
            )
        )
    return result


def chunk_test(
    data_dirs: list[Path],
    hyperparameter: SafetyDict,
    chunk_start: int,
    chunk_step: int,
    chunk_end: int,
    count: int = 1,
):
    return ct(
        data_dirs,
        hyperparameter,
        chunk_start,
        chunk_step,
        chunk_end,
        rt_whisper,
        whisper_streaming,
        count,
    )


def hyperparameter_test(
    test_step: int,
    data_dirs: list[Path],
    hyperparameter: dict,
    chunk_size: int,
    count: int = 1,
) -> dict:
    return ht(
        test_step,
        data_dirs,
        hyperparameter,
        chunk_size,
        rt_whisper,
        count,
    )


__all__ = [
    "whisper",
    "rt_whisper",
    "whisper_streaming",
    "chunk_test",
    "hyperparameter_test",
    "generate_statistical_dict",
    "generate_statistical_list",
    "show_table",
    "show_ribbon_plot",
    "show_ribbon_plot_with_mm",
    "show_line_plot",
]
