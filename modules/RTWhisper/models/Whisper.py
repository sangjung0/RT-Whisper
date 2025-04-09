from typing import Union
from faster_whisper import WhisperModel, BatchedInferencePipeline
from whisper import tokenizer
import numpy as np

from RTWhisper import BaseObject
from RTWhisper import Settings

class Whisper(BaseObject):
  def __init__(
    self, 
    beam_size:int = Settings.MODEL_BEAM_SIZE, 
    batch_size:int = Settings.MODEL_BATCH_SIZE, 
    model_size:str = Settings.MODEL_SIZE,
    device:str = Settings.MODEL_DEVICE,
    compute_type:str = Settings.MODEL_COMPUTE_TYPE
  ):
    super().__init__()
    self.__BEAM_SIZE = beam_size
    self.__BATCH_SIZE = batch_size

    self.__model = WhisperModel(model_size, device=device, compute_type=compute_type)
    self.__batched_model = BatchedInferencePipeline(self.__model)
    self.__tokenizer = tokenizer.get_tokenizer(multilingual=True)

  def transcribe(self, audio: np.ndarray, language:str = None, prompt: str = ""):
    return self.__model.transcribe(
      audio, beam_size=self.__BEAM_SIZE, language=language,
      word_timestamps=True, vad_filter=False,
      initial_prompt=prompt
    )

  def batched_transcribe(self, audio: Union[np.ndarray, list[np.ndarray]], language:str = None):
    if isinstance(audio, list) and len(audio) > self.__BATCH_SIZE:
      raise ValueError("The maximum number of audio files is 8.")

    return self.__batched_model.transcribe(
      audio, batch_size=self.__BATCH_SIZE, 
      beam_size=self.__BEAM_SIZE, language=language,
      word_timestamps=True, vad_filter=False
    )

  @property
  def tokenizer(self) -> tokenizer.Tokenizer:
    return self.__tokenizer