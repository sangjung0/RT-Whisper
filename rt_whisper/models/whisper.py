from faster_whisper import WhisperModel
import numpy as np

from sj_utils.decorator_utils import singleton


@singleton
class Whisper:
    def __init__(self, options: dict = {}):
        super().__init__()
        self._model = WhisperModel(**options)

    def transcribe(
        self, audio: np.ndarray, language: str, prompt: str, options: dict = {}
    ):
        return self._model.transcribe(
            audio,
            language=language,
            **options,
            initial_prompt=prompt,
            word_timestamps=True,
        )

    @staticmethod
    @property
    def sample_rate():
        return 16000  # WhisperModel의 샘플레이트는 16000Hz로 고정되어 있음
