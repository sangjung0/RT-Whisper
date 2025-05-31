from faster_whisper import WhisperModel
import numpy as np
from whisper import tokenizer

from RTWhisper import BaseObject
from RTWhisper import Settings
from RTWhisper import Pipeline

from RTWhisper.data import Context


class Whisper(BaseObject, Pipeline):
    def __init__(
        self,
        beam_size: int = Settings.MODEL_BEAM_SIZE,
        model_size: str = Settings.MODEL_SIZE,
        device: str = Settings.MODEL_DEVICE,
        compute_type: str = Settings.MODEL_COMPUTE_TYPE,
        SAMPLE_RATE: int = Settings.MODEL_SAMPLE_RATE,
    ):
        super().__init__()
        self._BEAM_SIZE = beam_size
        self._SAMPLE_RATE = SAMPLE_RATE

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.__tokenizer = tokenizer.get_tokenizer(multilingual=True)

    @property
    def tokenizer(self) -> tokenizer.Tokenizer:
        return self.__tokenizer

    def transcribe(self, audio: np.ndarray, language: str, prompt: str):
        return self._model.transcribe(
            audio,
            beam_size=self._BEAM_SIZE,
            language=language,
            word_timestamps=True,
            vad_filter=False,
            initial_prompt=prompt,
        )

    def can_process(self, context: Context) -> bool:
        if context.merged_processed_audio_sc == 0:
            return False
        return context.merged_processed_audio, context.language, context.prompt

    def compute_process(self, param: tuple) -> tuple:
        audio, language, prompt = param
        segments, info = self.transcribe(audio, language, prompt)
        return segments, info.language

    def apply_process(self, context: Context, result: tuple) -> None:
        segments, language = result
        context.merged_candidate_tokens = segments
        context.language = language
