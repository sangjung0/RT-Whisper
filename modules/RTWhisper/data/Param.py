from typing import Any
import numpy as np

from .Token import Token
from .Sentence import Sentence
from .Result import Result

STATISTIC = {
  "probability": {
    "mean": {
    },
    "std": {
    },
    "count": {
    },
  },
  "duration": {
    "mean": {
    },
    "std": {
    },
    "count": {
    },
  }
}

class Param:
  def __init__(
    self,
    audio:np.ndarray = None,
    order: int = 0,
    sc_offset: int= 0,
    prev_audio_sc: int = 0,
    prev_processed_audio:np.ndarray = None,
    prev_timestamps: list[dict] = [],
    prev_sentence:Sentence = None,
    prev_words:list[Token] = [],
    prev_recog:list[Token] = [],
    statistics:dict[str : dict[str : float]] = STATISTIC,
    prompt:str = None,
    language: str = None,
  ):
    self.__audio = audio
    self.__order = order
    self.__sc_offset = sc_offset
    self.__prev_audio_sc = prev_audio_sc
    self.__prev_processed_audio = prev_processed_audio
    self.__prev_timestamps = prev_timestamps
    self.__prev_sentence = prev_sentence
    self.__prev_words = prev_words
    self.__prev_recog = prev_recog
    self.__statistics = statistics
    self.__prompt = prompt
    self.__language = language

  def update(self, result:Result = None):
    self.__order = result.order
    self.__sc_offset = result.sc_offset
    self.__prev_audio_sc = result.prev_audio_sc
    self.__prev_processed_audio = result.prev_processed_audio
    self.__prev_timestamps = result.prev_timestamps
    self.__prev_sentence = result.prev_sentence
    self.__prev_words = result.prev_words
    self.__prev_recog = result.prev_recog
    self.__statistics = result.statistics


  def to_dict(self):
    return {
      "order": self.__order,
      "sc_offset": self.__sc_offset,
      "prev_audio_sc": self.__prev_audio_sc,
      "prev_timestamps": self.__prev_timestamps,
      "prev_sentence": self.__prev_sentence.to_dict() if self.__prev_sentence else None,
      "prev_words": [word.to_dict() for word in self.__prev_words],
      "prev_recog": [word.to_dict() for word in self.__prev_recog],
      "statistics": self.__statistics,
      "prompt": self.__prompt,
      "language": self.__language,
    }

  @classmethod
  def from_dict(cls, d: dict[str, Any]):
    return cls(
      audio=d.get("audio", None),
      order=d.get("order", 0),
      sc_offset=d.get("sc_offset", 0),
      prev_audio_sc=d.get("prev_audio_sc", 0),
      prev_processed_audio=d.get("prev_processed_audio", None),
      prev_timestamps=d.get("prev_timestamps", []),
      prev_sentence=Sentence.from_dict(d.get("prev_sentence", {})) if d.get("prev_sentence") else None,
      prev_words=[Token.from_dict(w) for w in d.get("prev_words", [])],
      prev_recog=[Token.from_dict(w) for w in d.get("prev_recog", [])],
      statistics= d.get("statistics", STATISTIC),
      prompt=d.get("prompt", None),
      language=d.get("language", None),
    )
    
  def __json__(self):
    return self.to_dict()

  @property
  def audio(self):
    return self.__audio
  @audio.setter
  def audio(self, value:np.ndarray):
    if not isinstance(value, np.ndarray) and value is not None:
      raise TypeError("Audio must be a numpy array")
    self.__audio = value

  @property
  def order(self):
    return self.__order
  @order.setter
  def order(self, value:int):
    if not isinstance(value, int):
      raise TypeError("Order must be an integer")
    self.__order = value

  @property
  def sc_offset(self):
    return self.__sc_offset
  @sc_offset.setter
  def sc_offset(self, value:float):
    if not (isinstance(value, float) or isinstance(value, int)):
      raise TypeError("Time offset must be a float")
    self.__sc_offset = value

  @property
  def prev_audio_sc(self):
    return self.__prev_audio_sc
  @prev_audio_sc.setter
  def prev_audio_sc(self, value:int):
    if not isinstance(value, int):
      raise TypeError("Previous audio sample count must be an integer")
    self.__prev_audio_sc = value

  @property
  def prev_processed_audio(self):
    return self.__prev_processed_audio
  @prev_processed_audio.setter
  def prev_audio(self, value:np.ndarray):
    if not isinstance(value, np.ndarray) and value is not None:
      raise TypeError("Previous audio must be a numpy array")
    self.__prev_processed_audio = value

  @property
  def prev_timestamps(self):
    return self.__prev_timestamps
  @prev_timestamps.setter
  def prev_timestamps(self, value:list[dict]):
    if not isinstance(value, list):
      raise TypeError("Previous timestamps must be a list")
    self.__prev_timestamps = value

  @property
  def prev_sentence(self):
    return self.__prev_sentence
  @prev_sentence.setter
  def prev_sentence(self, value:Sentence):
    if not isinstance(value, Sentence) and value is not None:
      raise TypeError("Previous sentence must be a Sentence instance")
    self.__prev_sentence = value

  @property
  def prev_words(self):
    return self.__prev_words
  @prev_words.setter
  def prev_words(self, value:list[Token]):
    if not isinstance(value, list):
      raise TypeError("Previous words must be a list")
    if not all(isinstance(w, Token) for w in value):
      raise TypeError("All elements in previous words must be Token instances")
    self.__prev_words = value
    
  @property
  def prev_recog(self):
    return self.__prev_recog
  @prev_recog.setter
  def prev_recog(self, value:list[Token]):
    if not isinstance(value, list):
      raise TypeError("Previous recognition must be a list")
    if not all(isinstance(w, Token) for w in value):
      raise TypeError("All elements in previous recognition must be Token instances")
    self.__prev_recog = value
    
  @property
  def statistics(self):
    return self.__statistics
  @statistics.setter
  def statistics(self, value:dict[str : dict[str : float]]):
    if not isinstance(value, dict):
      raise TypeError("Statistics must be a dictionary")
    self.__statistics = value

  @property
  def prompt(self):
    return self.__prompt
  @prompt.setter
  def prompt(self, value:str):
    if not isinstance(value, str) and value is not None:
      raise TypeError("Prompt must be a string")
    self.__prompt = value

  @property
  def language(self):
    return self.__language
  @language.setter
  def language(self, value:str):
    if not isinstance(value, str) and value is not None:
      raise TypeError("Language must be a string")
    self.__language = value