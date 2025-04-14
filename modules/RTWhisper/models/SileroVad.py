import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

from RTWhisper import BaseObject, Pipeline, Settings
from RTWhisper.data import Context


class SileroVad(BaseObject, Pipeline):
  def __init__(
    self,
    SAMPLE_RATE:int = Settings.MODEL_SAMPLE_RATE,
  ):
    super().__init__()
    self.__SAMPLE_RATE = SAMPLE_RATE
    self.__model = load_silero_vad(onnx=True)

  def process(self, context:Context):
    audio = context.processed_audio

    if len(audio) == 0:
      return

    timestamps = get_speech_timestamps(
      audio,
      self.__model,
      sampling_rate=self.__SAMPLE_RATE,
      threshold = 0.4,
      min_silence_duration_ms = 400,
      speech_pad_ms = 300
    )

    merged_audio = []

    for segment in timestamps:
      start = segment['start']
      end = segment['end']
      merged_audio.append(audio[start:end])

    if merged_audio:
      context.processed_audio = np.concatenate(merged_audio)
    else:
      context.processed_audio = np.zeros((0,), dtype=np.float32)
    context.timestamps = timestamps
      
