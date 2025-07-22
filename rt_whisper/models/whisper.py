from faster_whisper import WhisperModel
import numpy as np

from sj_utils.decorator import singleton


@singleton
class Whisper:
    SAMPLE_RATE = 16000

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
