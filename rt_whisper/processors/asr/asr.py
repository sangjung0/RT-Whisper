from typing import Any, Callable, Iterable

import numpy as np
from .asr_recycler import ASRRecycler


class ASR(ASRRecycler):
    def __init__(
        self,
        *args,
        transcriber: Callable[[np.ndarray, str, str], tuple[Iterable, Any]],
        tokenizer_encode: Callable[[str], list[int]],
        sample_rate: int,
        within_eos: bool,
        **kwargs
    ):
        super().__init__(
            *args,
            transcriber=transcriber,
            tokenizer_encode=tokenizer_encode,
            sample_rate=sample_rate,
            within_eos=within_eos,
            **kwargs
        )
