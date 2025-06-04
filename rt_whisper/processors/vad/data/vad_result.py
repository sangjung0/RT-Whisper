from dataclasses import dataclass
import numpy as np

from rt_whisper.data import Context


@dataclass(slots=True)
class VadResult:
    vad_chunk: np.ndarray
    vad_timestamps: list[dict[str, int]]

    def update_context(self, context: Context) -> None:
        context.vad_chunk = self.vad_chunk
        context.vad_timestamps = self.vad_timestamps
