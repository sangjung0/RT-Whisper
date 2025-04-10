import numpy as np 

from .Sentence import Sentence
from .Token import Token

class Result:
  def __init__(
    self,
    completed:dict[int, Sentence] = {},
    order: int = 0,
    sc_offset: float = 0,
    processed_audio:np.ndarray = None,
    prev_audio_sc: int = 0,
    prev_processed_audio:np.ndarray = None,
    prev_timestamps: list[dict] = [],
    prev_sentence:Sentence = None,
    prev_words:list[Token] = [],
    prev_recog:list[Token] = [],
    statistics:dict[str : dict[str : float]] = {},
  ): 
    self.completed = completed
    self.__order = order
    self.__sc_offset = sc_offset
    self.processed_audio = processed_audio
    self.__prev_audio_sc = prev_audio_sc
    self.prev_processed_audio = prev_processed_audio
    self.prev_timestamps = prev_timestamps
    self.prev_sentence = prev_sentence
    self.__prev_words = tuple(prev_words) if prev_words else tuple()
    self.__prev_recog = tuple(prev_recog) if prev_recog else tuple()
    self.statistics = statistics

  @property
  def order(self):
    return self.__order
  
  @property
  def sc_offset(self):
    return self.__sc_offset

  @property
  def prev_audio_sc(self):
    return self.__prev_audio_sc

  @property
  def prev_words(self):
    return list(self.__prev_words)

  @property
  def prev_recog(self):
    return list(self.__prev_recog)