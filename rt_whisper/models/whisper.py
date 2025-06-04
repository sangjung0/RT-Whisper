from faster_whisper import WhisperModel
import numpy as np
from whisper import tokenizer

from rt_whisper.abstracts import Singleton
from rt_whisper.core.state import config


class Whisper(Singleton):
    def __init__(
        self,
        model_size: str = config.rt_whisper.model_size,
        device: str = config.rt_whisper.model_device,
        compute_type: str = config.rt_whisper.model_compute_type,
        beam_size: int = config.rt_whisper.model_beam_size,
    ):
        super().__init__()
        self.__BEAM_SIZE = beam_size

        self._model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def transcribe(self, audio: np.ndarray, language: str, prompt: str):
        return self._model.transcribe(
            audio,
            beam_size=self.__BEAM_SIZE,
            language=language,
            word_timestamps=True,
            vad_filter=False,
            initial_prompt=prompt,
        )

    def get_tokenizer(self):
        return tokenizer.get_tokenizer(multilingual=True)
