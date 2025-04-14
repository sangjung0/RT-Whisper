import numpy as np

from .Sentence import Sentence
from .Token import Token
from .Param import Param
from .Result import Result

class Context:
  def __init__(self):
    # need update
    self.order:int = 0 # order of the completed sentence
    self.sc_offset:float = 0 # sample count offset
    self.audio:np.ndarray = np.zeros((0,), dtype=np.float32) # raw audio
    self.prev_audio_sc:int = 0 # sample rate of prev audio
    self.prev_processed_audio:np.ndarray = np.zeros((0,), dtype=np.float32)
    self.prev_timestamps: list[dict] = []
    self.prev_sentence:Sentence = None
    self.prev_words:list[Token] = []
    self.prev_recog:list[Token] = []
    self.statistics:dict[str : dict[str : float]] = {}
    self.prompt:str = None
    self.language: str = None

    # don't need update
    self.processed_audio:np.ndarray = np.zeros((0,), dtype=np.float32) # pre processed audio
    self.timestamps: list[dict] = [] # timestamps
    self.tokens:list[Token] = [] 
    self.processed_timestamp_conditions: list[tuple[int, int, int]] = [] # processed timestamp conditions
    self.completed_words:list[Token] = []
    self.completed:dict[int, Sentence] = {}

  def bind(self, param:Param):
    self.order = param.order
    self.sc_offset = param.sc_offset
    self.audio = self.processed_audio = param.audio
    self.timestamps = [] if len(param.audio) == 0 else [{"start": 0, "end": len(param.audio)}]
    self.prev_audio_sc = param.prev_audio_sc
    self.prev_processed_audio = param.prev_processed_audio
    self.prev_timestamps = param.prev_timestamps
    self.prev_sentence = param.prev_sentence
    self.prev_words = param.prev_words
    self.prev_recog = param.prev_recog
    self.statistics = param.statistics
    self.prompt = param.prompt
    self.language = param.language

  def extract(self):
    return Result(
      completed=self.completed,
      order=self.order,
      sc_offset = self.sc_offset,
      processed_audio = self.processed_audio,
      prev_audio_sc  = self.prev_audio_sc,
      prev_processed_audio=self.prev_processed_audio,
      prev_timestamps=self.prev_timestamps,
      prev_sentence=self.prev_sentence,
      prev_words=self.prev_words,
      prev_recog=self.prev_recog,
      statistics = self.statistics
    )

  