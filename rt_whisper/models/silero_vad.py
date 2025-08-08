import numpy as np

from silero_vad import load_silero_vad, get_speech_timestamps

from sj_utils.decorator import singleton


@singleton
class SileroVad:
    def __init__(self, sample_rate: int, options: dict = {}):
        super().__init__()
        self.__model = load_silero_vad(**options, onnx=True)
        self.__SAMPLE_RATE = sample_rate

    def run(self, audio: np.ndarray, options: dict = {}) -> list[dict[str, int]]:
        return get_speech_timestamps(
            audio, self.__model, sampling_rate=self.__SAMPLE_RATE, **options
        )


__all__ = ["SileroVad"]
