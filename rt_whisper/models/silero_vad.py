import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

from sj_utils.decorator_utils import singleton

from rt_whisper.core.state import config


@singleton
class SileroVad:
    def __init__(self, sample_rate: int = config.rt_whisper.model_sample_rate):
        super().__init__()
        self.__model = load_silero_vad(onnx=True)
        self.__SAMPLE_RATE = sample_rate

    def run(self, audio: np.ndarray) -> list[dict[str, int]]:
        return get_speech_timestamps(
            audio,
            self.__model,
            sampling_rate=self.__SAMPLE_RATE,
            threshold=0.45,
            speech_pad_ms=200,
        )
