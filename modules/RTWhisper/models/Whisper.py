from faster_whisper import WhisperModel
from whisper import tokenizer

from RTWhisper import BaseObject
from RTWhisper import Settings
from RTWhisper import Pipeline

from RTWhisper.data import Context, Token


class Whisper(BaseObject, Pipeline):
  def __init__(
    self, 
    beam_size:int = Settings.MODEL_BEAM_SIZE, 
    model_size:str = Settings.MODEL_SIZE,
    device:str = Settings.MODEL_DEVICE,
    compute_type:str = Settings.MODEL_COMPUTE_TYPE,
    SAMPLE_RATE:int = Settings.MODEL_SAMPLE_RATE,
  ):
    super().__init__()
    self._BEAM_SIZE = beam_size
    self._SAMPLE_RATE = SAMPLE_RATE

    self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
    self.__tokenizer = tokenizer.get_tokenizer(multilingual=True)

  def process(self, context:Context) -> None:
    audio = context.processed_audio
    language = context.language
    prompt = context.prompt

    if audio is None: return

    segments, info = self._model.transcribe(
      audio, beam_size=self._BEAM_SIZE, language=language,
      word_timestamps=True, vad_filter=False,
      initial_prompt=prompt
    )

    context.tokens = segments
    context.language = info.language

  @property
  def tokenizer(self) -> tokenizer.Tokenizer:
    return self.__tokenizer