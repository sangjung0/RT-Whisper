import math
import re

import numpy as np


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_empty_chunk(dtype=np.float32) -> np.ndarray:
    return np.zeros((0,), dtype=dtype)


def update_mean_std(
    old_mean: float,
    old_std: float,
    old_count: int,
    new_mean: float,
    new_std: float,
    new_count: int,
) -> tuple[float, float]:
    if old_count == 0:
        return new_mean, new_std
    if new_count == 0:
        return old_mean, old_std

    total_count = old_count + new_count
    updated_mean = (old_mean * old_count + new_mean * new_count) / total_count

    updated_std = math.sqrt(
        (
            old_count * (old_std**2 + (old_mean - updated_mean) ** 2)
            + new_count * (new_std**2 + (new_mean - updated_mean) ** 2)
        )
        / total_count
    )

    return updated_mean, updated_std
