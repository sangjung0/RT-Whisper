from __future__ import annotations
from typing import TYPE_CHECKING

import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/test/performance_test",
    "/workspaces/dev/test/performance_test/esic",
]
for path in paths:
    sys.path.append(os.path.abspath(path))

import numpy as np

from typing import Any
from matplotlib import pyplot as plt
from pathlib import Path

from sj_utils.collection import SafetyDict
from sj_utils.statistics import summarize_distribution
from sj_ai_utils.datasets import Dataset

from common_util import rt_whisper as rtw, whisper_streaming as ws, whisper as w

if TYPE_CHECKING:
    pass


def whisper(dataset: Any, repeat: int = 1) -> list[dict]:
    result = []
    for _ in range(repeat):
        result.append(w(dataset))
    return result


def rt_whisper(
    dataset: Dataset,
    hyperparameter: SafetyDict,
    chunk_size: int,
    repeat: int = 1,
) -> list[dict]:
    result = []
    for _ in range(repeat):
        result.append(
            rtw(
                Path("/"),
                dataset,
                use_save_loader=False,
                use_prompt=True,
                hyperparameter=hyperparameter,
                chunk_size=chunk_size,
            )
        )
    return result


def whisper_streaming(
    dataset: Dataset,
    chunk_size: int,
    repeat: int = 1,
) -> list[dict]:
    result = []
    for _ in range(repeat):
        result.append(ws(dataset, chunk_size=chunk_size))
    return result


def chunk_test(
    dataset: Dataset,
    hyperparameter: SafetyDict,
    chunk_start: int,
    chunk_step: int,
    chunk_end: int,
):
    result = {}
    for chunk_size in range(chunk_start, chunk_end + 1, chunk_step):
        result[chunk_size] = {
            "rt_whisper": rt_whisper(dataset, hyperparameter, chunk_size, repeat=1)[0],
            "whisper_streaming": whisper_streaming(dataset, chunk_size, repeat=1)[0],
        }
    return result


def max_overlap_duration_test(
    test_step: int,
    dataset: Dataset,
    hyperparameter: dict,
    chunk_size: int,
):
    result = {}
    original_value = hyperparameter["asr"]["max_overlap_duration"]
    start = original_value - 16000 * test_step
    for od in range(max(0, start), original_value + 16000 * test_step, 16000):
        hyperparameter["asr"]["max_overlap_duration"] = od
        result[od] = rt_whisper(
            dataset, SafetyDict(hyperparameter), chunk_size, repeat=1
        )[0]
    hyperparameter["asr"]["max_overlap_duration"] = original_value
    return result


def max_prompt_words_test(
    test_step: int,
    dataset: Dataset,
    hyperparameter: dict,
    chunk_size: int,
):
    result = {}
    original_value = hyperparameter["asr"]["max_prompt_words"]
    start = original_value - test_step
    for mpw in range(max(0, start), original_value + test_step, 1):
        hyperparameter["asr"]["max_prompt_words"] = mpw
        result[mpw] = rt_whisper(
            dataset, SafetyDict(hyperparameter), chunk_size, repeat=1
        )[0]
    hyperparameter["asr"]["max_prompt_words"] = original_value
    return result


def boundary_test(
    test_step: int,
    dataset: Dataset,
    hyperparameter: SafetyDict,
    chunk_size: int,
):
    result = {}
    original_value = hyperparameter["position_weighted_filter"]["boundary"]
    start = original_value - test_step * 160
    for bd in range(max(0, start), original_value + test_step * 160, 160):
        hyperparameter["position_weighted_filter"]["boundary"] = bd
        result[bd] = rt_whisper(
            dataset, SafetyDict(hyperparameter), chunk_size, repeat=1
        )[0]
    hyperparameter["position_weighted_filter"]["boundary"] = original_value
    return result


def min_duration_test(
    test_step: int,
    dataset: Dataset,
    hyperparameter: dict,
    chunk_size: int,
):
    result = {}
    original_value = hyperparameter["duration_filter"]["min_dur"]["en"]
    start = original_value - test_step * 160
    for md in range(max(0, start), original_value + test_step * 160, 160):
        hyperparameter["duration_filter"]["min_dur"]["en"] = md
        result[md] = rt_whisper(
            dataset, SafetyDict(hyperparameter), chunk_size, repeat=1
        )[0]
    hyperparameter["duration_filter"]["min_dur"]["en"] = original_value
    return result


def min_probability_test(
    test_step: int,
    dataset: Dataset,
    hyperparameter: dict,
    chunk_size: int,
):
    result = {}
    original_value = hyperparameter["probability_filter"]["min_prob"]["en"]
    start = original_value - test_step * 0.05
    for mp in np.arange(max(0, start), original_value + test_step * 0.05, 0.05):
        hyperparameter["probability_filter"]["min_prob"]["en"] = mp
        result[mp] = rt_whisper(
            dataset, SafetyDict(hyperparameter), chunk_size, repeat=1
        )[0]
    hyperparameter["probability_filter"]["min_prob"]["en"] = original_value
    return result


def iou_threshold_test(
    test_step: int,
    dataset: Dataset,
    hyperparameter: dict,
    chunk_size: int,
):
    result = {}
    original_value = hyperparameter["selector"]["iou_threshold"]["en"]
    start = original_value - test_step * 0.05
    for iou in np.arange(max(0, start), original_value + test_step * 0.05, 0.05):
        hyperparameter["selector"]["iou_threshold"]["en"] = iou
        result[iou] = rt_whisper(
            dataset, SafetyDict(hyperparameter), chunk_size, repeat=1
        )[0]
    hyperparameter["selector"]["iou_threshold"]["en"] = original_value
    return result


def cos_threshold_test(
    test_step: int,
    dataset: Dataset,
    hyperparameter: dict,
    chunk_size: int,
):
    result = {}
    original_value = hyperparameter["selector"]["cos_threshold"]["en"]
    start = original_value - test_step * 0.05
    for cos in np.arange(max(0, start), original_value + test_step * 0.05, 0.05):
        hyperparameter["selector"]["cos_threshold"]["en"] = cos
        result[cos] = rt_whisper(
            dataset, SafetyDict(hyperparameter), chunk_size, repeat=1
        )[0]
    hyperparameter["selector"]["cos_threshold"]["en"] = original_value
    return result


def padding_test(
    test_step: int,
    dataset: Dataset,
    hyperparameter: dict,
    chunk_size: int,
):
    result = {}
    original_value = hyperparameter["selector"]["padding"]["en"]
    start = original_value - test_step * 160
    for padding in range(max(0, start), original_value + test_step * 160, 160):
        hyperparameter["selector"]["padding"]["en"] = padding
        result[padding] = rt_whisper(
            dataset, SafetyDict(hyperparameter), chunk_size, repeat=1
        )[0]
    hyperparameter["selector"]["padding"]["en"] = original_value
    return result


def hyperparameter_test(
    test_step: int,
    dataset: Dataset,
    hyperparameter: dict,
    chunk_size: int,
):
    return {
        "max_overlap_duration": max_overlap_duration_test(
            test_step, dataset, hyperparameter, chunk_size
        ),
        "max_prompt_words": max_prompt_words_test(
            test_step, dataset, hyperparameter, chunk_size
        ),
        "boundary": boundary_test(test_step, dataset, hyperparameter, chunk_size),
        "min_duration": min_duration_test(
            test_step, dataset, hyperparameter, chunk_size
        ),
        "min_probability": min_probability_test(
            test_step, dataset, hyperparameter, chunk_size
        ),
        "iou_threshold": iou_threshold_test(
            test_step, dataset, hyperparameter, chunk_size
        ),
        "cos_threshold": cos_threshold_test(
            test_step, dataset, hyperparameter, chunk_size
        ),
        "padding": padding_test(test_step, dataset, hyperparameter, chunk_size),
    }


def generate_statistical_list(data: list[dict]):
    result = {}
    keys = data[0].keys()
    for key in keys:
        statistic = summarize_distribution([d[key] for d in data])
        result[key] = f"{statistic['mean']:.2f}+-{statistic['std']:.2f}"
    return result


def generate_statistical_dict(data: list[list]):
    result = []
    idx = len(data[0])
    for i in range(idx):
        statistic = summarize_distribution([d[i] for d in data])
        result.append(f"{statistic['mean']:.2f}+-{statistic['std']:.2f}")
    return result


def show_table(labels: list[str], data: list[list], title: str = "Results"):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    table = ax.table(cellText=data, colLabels=labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title(title, fontsize=16, pad=10)
    plt.tight_layout()
    plt.show()


def show_ribbon_plot(
    x: list,
    m1_mean: list,
    m1_q1: list,
    m1_q2: list,
    m1_q3: list,
    m2_mean: list,
    m2_q1: list,
    m2_q2: list,
    m2_q3: list,
    title: str,
    x_label: str,
    y_label: str,
    m1_label: str,
    m2_label: str,
):
    plt.figure(figsize=(10, 6))

    # 모델 1
    plt.plot(x, m1_mean, marker="o", color="blue", label=f"{m1_label} Mean")
    plt.fill_between(
        x, m1_q1, m1_q2, color="lightblue", alpha=0.4, label=f"{m1_label} Q1–Q2"
    )
    plt.fill_between(
        x, m1_q2, m1_q3, color="skyblue", alpha=0.4, label=f"{m1_label} Q2–Q3"
    )

    # 모델 2
    plt.plot(x, m2_mean, marker="o", color="orange", label=f"{m2_label} Mean")
    plt.fill_between(
        x, m2_q1, m2_q2, color="navajowhite", alpha=0.4, label=f"{m2_label} Q1–Q2"
    )
    plt.fill_between(
        x, m2_q2, m2_q3, color="lightsalmon", alpha=0.4, label=f"{m2_label} Q2–Q3"
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_ribbon_plot_with_mm(
    x: list,
    m1_mean: list,
    m1_min: list,
    m1_q1: list,
    m1_q2: list,
    m1_q3: list,
    m1_max: list,
    m2_mean: list,
    m2_min: list,
    m2_q1: list,
    m2_q2: list,
    m2_q3: list,
    m2_max: list,
    title: str,
    x_label: str,
    y_label: str,
    m1_label: str,
    m2_label: str,
):
    plt.figure(figsize=(10, 6))

    # 모델 1
    plt.plot(x, m1_mean, marker="o", color="blue", label=f"{m1_label} Mean")
    plt.plot(x, m1_min, linestyle="--", color="darkblue", label=f"{m1_label} Min")
    plt.plot(x, m1_max, linestyle="--", color="crimson", label=f"{m1_label} Max")
    plt.fill_between(
        x, m1_q1, m1_q3, color="lightblue", alpha=0.4, label=f"{m1_label} Q1–Q3"
    )
    plt.fill_between(
        x, m1_q2, m1_q3, color="skyblue", alpha=0.4, label=f"{m1_label} Q2–Q3"
    )

    # 모델 2
    plt.plot(x, m2_mean, marker="o", color="orange", label=f"{m2_label} Mean")
    plt.plot(x, m2_min, linestyle="--", color="saddlebrown", label=f"{m2_label} Min")
    plt.plot(x, m2_max, linestyle="--", color="firebrick", label=f"{m2_label} Max")
    plt.fill_between(
        x, m2_q1, m2_q3, color="navajowhite", alpha=0.4, label=f"{m2_label} Q1–Q3"
    )
    plt.fill_between(
        x, m2_q2, m2_q3, color="lightsalmon", alpha=0.4, label=f"{m2_label} Q2–Q3"
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_line_plot(
    x: list,
    m1_y: list,
    m2_y: list,
    title: str,
    x_label: str,
    y_label: str,
    m1_label: str,
    m2_label: str,
):
    plt.figure(figsize=(10, 6))

    plt.plot(x, m1_y, marker="o", linestyle="-", color="blue", label=m1_label)
    plt.plot(x, m2_y, marker="s", linestyle="--", color="orange", label=m2_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


__all__ = [
    "whisper",
    "rt_whisper",
    "whisper_streaming",
    "chunk_test",
    "hyperparameter_test",
    "generate_statistical_list",
    "generate_statistical_dict",
    "show_table",
    "show_ribbon_plot",
    "show_ribbon_plot_with_mm",
    "show_line_plot",
]
