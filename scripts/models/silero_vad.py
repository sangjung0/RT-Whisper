import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

from core import BaseObject
from abstracts import Pipeline
from settings import Settings
from data import Context


class SileroVad(BaseObject, Pipeline):
    def __init__(
        self,
        SAMPLE_RATE: int = Settings.MODEL_SAMPLE_RATE,
    ):
        super().__init__()
        self.__SAMPLE_RATE = SAMPLE_RATE
        self.__model = load_silero_vad(onnx=True)

    def get_timestamps(self, audio: np.ndarray):
        return get_speech_timestamps(
            audio,
            self.__model,
            sampling_rate=self.__SAMPLE_RATE,
            #   threshold = 0.4,
            #   min_silence_duration_ms = 400,
            #   speech_pad_ms = 300
        )

    def can_process(self, context: Context) -> bool:
        if context.processed_audio_sc == 0:
            return False
        return context.processed_audio

    def compute_process(self, processed_audio):
        timestamps = self.get_timestamps(processed_audio)
        merged_audio = []
        for segment in timestamps:
            start = segment["start"]
            end = segment["end"]
            merged_audio.append(processed_audio[start:end])

        new_processed_audio = (
            np.concatenate(merged_audio)
            if merged_audio
            else np.zeros((0,), dtype=np.float32)
        )

        return new_processed_audio, timestamps

    def apply_process(self, context: Context, result: tuple):
        new_processed_audio, timestamps = result
        context.processed_audio = new_processed_audio
        context.timestamps = timestamps
