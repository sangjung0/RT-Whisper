from typing import Callable

import numpy as np
from .vad_recycler import VADRecycler


class VAD(VADRecycler):
    def __init__(
        self, vad: Callable[[np.ndarray], list[dict[str, int]]], *args, **kwargs
    ):
        super().__init__(vad, *args, **kwargs)
