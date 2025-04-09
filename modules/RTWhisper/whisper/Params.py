from logging import Logger
from typing import Any
import numpy as np

from RTWhisper import BaseLogger

from .Token import Token
from .Sentence import Sentence

class SafetyDict(dict):
  def __init__(self, default_value:Any, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._DEFAULT_VALUE = default_value
  
  def __getitem__(self, key):
    return super().get(key, self._DEFAULT_VALUE)


class WordParams:
  def __init__(
    self,
    audio:np.ndarray = None,
    order: int = 0,
    time_offset: float = 0,
    prev_audio:np.ndarray = None,
    prev_words:list[Token] = [],
    prev_recog:list[Token] = [],
    prev_prob_mean: dict[str:float] = {},
    prev_prob_std: dict[str:float] = {},
    prev_prob_count: dict[str:int] = {},
    prev_dura_mean: dict[str:float] = {},
    prev_dura_std: dict[str:float] = {},
    prev_dura_count: dict[str:int] = {},
    language: str = None,
  ):
    self.__audio = audio
    self.__order = order
    self.__time_offset = time_offset
    self.__prev_audio = prev_audio
    self.__prev_words = prev_words
    self.__prev_recog = prev_recog
    self.__prev_prob_mean = SafetyDict(None, prev_prob_mean)
    self.__prev_prob_std = SafetyDict(None, prev_prob_std)
    self.__prev_prob_count = SafetyDict(None, prev_prob_count)
    self.__prev_dura_mean = SafetyDict(None, prev_dura_mean)
    self.__prev_dura_std = SafetyDict(None, prev_dura_std)
    self.__prev_dura_count = SafetyDict(None, prev_dura_count)
    self.__language = language

  def to_dict(self):
    return {
      "order": self.__order,
      "time_offset": self.__time_offset,
      "prev_words": [word.to_dict() for word in self.__prev_words],
      "prev_recog": [word.to_dict() for word in self.__prev_recog],
      "prev_prob_mean": self.__prev_prob_mean,
      "prev_prob_std": self.__prev_prob_std,
      "prev_prob_count": self.__prev_prob_count,
      "prev_dura_mean": self.__prev_dura_mean,
      "prev_dura_std": self.__prev_dura_std,
      "prev_dura_count": self.__prev_dura_count,
      "language": self.__language,
    }

  @classmethod
  def from_dict(cls, d: dict[str, Any]):
    return cls(
      audio=d.get("audio", None),
      order=d.get("order", 0),
      time_offset=d.get("time_offset", 0.0),
      prev_audio=d.get("prev_audio", None),
      prev_words=[Token.from_dict(w) for w in d.get("prev_words", [])],
      prev_recog=[Token.from_dict(w) for w in d.get("prev_recog", [])],
      prev_prob_mean=d.get("prev_prob_mean", {}),
      prev_prob_std=d.get("prev_prob_std", {}),
      prev_prob_count=d.get("prev_prob_count", {}),
      prev_dura_mean=d.get("prev_dura_mean", {}),
      prev_dura_std=d.get("prev_dura_std", {}),
      prev_dura_count=d.get("prev_dura_count", {}),
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
  def time_offset(self):
    return self.__time_offset
  @time_offset.setter
  def time_offset(self, value:float):
    if not (isinstance(value, float) or isinstance(value, int)):
      raise TypeError("Time offset must be a float")
    self.__time_offset = value

  @property
  def prev_audio(self):
    return self.__prev_audio
  @prev_audio.setter
  def prev_audio(self, value:np.ndarray):
    if not isinstance(value, np.ndarray) and value is not None:
      raise TypeError("Previous audio must be a numpy array")
    self.__prev_audio = value

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
  def prev_prob_mean(self):
    return self.__prev_prob_mean
  @prev_prob_mean.setter
  def prev_prob_mean(self, value:dict[str:float]):
    if not isinstance(value, dict):
      raise TypeError("Previous probability mean must be a dictionary")
    self.__prev_prob_mean = SafetyDict(None, value)

  @property
  def prev_prob_std(self):
    return self.__prev_prob_std
  @prev_prob_std.setter
  def prev_prob_std(self, value:dict[str:float]):
    if not isinstance(value, dict):
      raise TypeError("Previous probability std must be a dictionary")
    self.__prev_prob_std = SafetyDict(None, value)

  @property
  def prev_prob_count(self):
    return self.__prev_prob_count
  @prev_prob_count.setter
  def prev_prob_count(self, value:dict[str:int]):
    if not isinstance(value, dict):
      raise TypeError("Previous probability count must be a dictionary")
    self.__prev_prob_count = SafetyDict(None, value)

  @property
  def prev_dura_mean(self):
    return self.__prev_dura_mean
  @prev_dura_mean.setter
  def prev_dura_mean(self, value:dict[str:float]):
    if not isinstance(value, dict):
      raise TypeError("Previous duration mean must be a dictionary")
    self.__prev_dura_mean = SafetyDict(None, value)
  
  @property
  def prev_dura_std(self):
    return self.__prev_dura_std
  @prev_dura_std.setter
  def prev_dura_std(self, value:dict[str:float]):
    if not isinstance(value, dict):
      raise TypeError("Previous duration std must be a dictionary")
    self.__prev_dura_std = SafetyDict(None, value)
    
  @property
  def prev_dura_count(self):
    return self.__prev_dura_count
  @prev_dura_count.setter
  def prev_dura_count(self, value:dict[str:int]):
    if not isinstance(value, dict):
      raise TypeError("Previous duration count must be a dictionary")
    self.__prev_dura_count = SafetyDict(None, value)

  @property
  def language(self):
    return self.__language
  @language.setter
  def language(self, value:str):
    if not isinstance(value, str) and value is not None:
      raise TypeError("Language must be a string")
    self.__language = value
  

class SentenceParams:
  def __init__(
    self,
    audio:np.ndarray = None,
    order: int = 0,
    time_offset: float = 0,
    prev_audio:np.ndarray = None,
    prev_sentence:Sentence = None,
    prompt:str = None,
    language:str = None,
  ):
    self.__audio = audio
    self.__order = order
    self.__time_offset = time_offset
    self.__prev_audio = prev_audio
    self.__prev_sentence = prev_sentence
    self.__prompt = prompt
    self.__language = language

  def to_dict(self):
    return {
      "order": self.__order,
      "time_offset": self.__time_offset,
      "prev_sentence": (
          self.__prev_sentence.to_dict() if self.__prev_sentence else None
      ),
      "prompt": self.__prompt,
      "language": self.__language,
    }

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> "SentenceParams":
      return cls(
          order=d.get("order", 0),
          time_offset=d.get("time_offset", 0.0),
          prev_sentence=(
              Sentence.from_dict(d["prev_sentence"]) 
              if d.get("prev_sentence", None) else None
          ),
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
  def time_offset(self):
    return self.__time_offset
  @time_offset.setter
  def time_offset(self, value:float):
    if not (isinstance(value, float) or isinstance(value, int)):
      raise TypeError("Time offset must be a float")
    self.__time_offset = value

  @property
  def prev_audio(self):
    return self.__prev_audio
  @prev_audio.setter
  def prev_audio(self, value:np.ndarray):
    if not isinstance(value, np.ndarray) and value is not None:
      raise TypeError("Previous audio must be a numpy array")
    self.__prev_audio = value

  @property
  def prev_sentence(self):
    return self.__prev_sentence
  @prev_sentence.setter
  def prev_sentence(self, value:Sentence):
    if not isinstance(value, Sentence) and value is not None:
      raise TypeError("Previous sentence must be a Sentence instance")
    self.__prev_sentence = value

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


class WordReturn:
  def __init__(
    self,
    completed_dict:dict[int, Sentence] = {},
    order: int = 0,
    time_offset: float = 0,
    prev_audio:np.ndarray = None,
    prev_words:list[Token] = [],
    prev_recog:list[Token] = [],
    prev_prob_mean: dict[str:float] = {},
    prev_prob_std: dict[str:float] = {},
    prev_prob_count: dict[str:int] = {},
    prev_dura_mean: dict[str:float] = {},
    prev_dura_std: dict[str:float] = {},
    prev_dura_count: dict[str:int] = {},
  ):
    self.completed_dict = SafetyDict(None, completed_dict)
    self.__order = order
    self.__time_offset = time_offset
    self.prev_audio = prev_audio
    self.__prev_words = tuple(prev_words) if prev_words else tuple()
    self.__prev_recog = tuple(prev_recog) if prev_recog else tuple()
    self.prev_prob_mean = SafetyDict(None, prev_prob_mean)
    self.prev_prob_std = SafetyDict(None, prev_prob_std)
    self.prev_prob_count = SafetyDict(None, prev_prob_count)
    self.prev_dura_mean = SafetyDict(None, prev_dura_mean)
    self.prev_dura_std = SafetyDict(None, prev_dura_std)
    self.prev_dura_count = SafetyDict(None, prev_dura_count)

  @property
  def order(self):
    return self.__order
  
  @property
  def time_offset(self):
    return self.__time_offset

  @property
  def prev_words(self):
    return list(self.__prev_words)

  @property
  def prev_recog(self):
    return list(self.__prev_recog)


class SentenceReturn:
  def __init__(
    self,
    completed_dict:dict[int, Sentence] = {},
    order: int = 0,
    time_offset: float = 0,
    prev_audio:np.ndarray = None,
    prev_sentence:Sentence = None,
    prev_recog:list[Sentence] = [],
  ):
    self.completed_dict = SafetyDict(None, completed_dict)
    self.__order = order
    self.__time_offset = time_offset
    self.prev_audio = prev_audio
    self.prev_sentence = prev_sentence
    self.__prev_recog = tuple(prev_recog) if prev_recog else tuple()

  @property
  def order(self):
    return self.__order
  @property
  def time_offset(self):
    return self.__time_offset
  @property
  def prev_recog(self):
    return list(self.__prev_recog)
  


HYPERPARAMETERS = {
  "weighted_prob_boundary": 0.3,
  "filter_by_duration_z": {
    "default": 2.0,
    "ko": 2.0,
    "en": 2.0,
  },
  "filter_by_probability": {
    "z": {
      "default": 2.0,
      "ko": 2.0,
      "en": 2.0,
    },
    "min_prob": {
      "default": 1.0,
      "ko": 0.4,
      "en": 0.4,
    },
  },
  "token_iou_padding": 0.2,
  "combine": {
    "search_range_time": {
      "default": 1.5,
      "ko": 1.5,
      "en": 1.5,
    },
    "threshold": {
      "default": 0.5,
      "ko": 0.25,
      "en": 0.5,
    },
    "tolerance": {
      "default": 0.3,
      "ko": 0.3,
      "en": 0.3,
    },
  },
  "refine_tolerance": {
    "default": 0.5,
    "ko": 0.5,
    "en": 0.5,
  }
}

class Hyperparameters(SafetyDict):
  @BaseLogger.object
  def __init__(
    self, 
    default_value:Any = None, 
    setting:dict = HYPERPARAMETERS, 
    base_logger:Logger = None, 
    *args, 
    **kwargs
  ):
    super().__init__(default_value, *args, **kwargs)
    self.__logger = base_logger
    if setting: self.update(self.__upload(setting))

  def __getitem__(self, key):
    if key not in self:
      self.__logger.warning(f"Key '{key}' not found in SafetyDict. Returning default value.")
    return super().get(key, self._DEFAULT_VALUE)

  def __setitem__(self, key, value):
    raise NotImplementedError("SafetyDict is read-only")
    
  def __upload(self, d:dict):
    default = d["default"] if "default" in d else None
    bucket = Hyperparameters(default, setting=None)
    for key, value in d.items():
      if isinstance(value, dict):
        super(Hyperparameters, bucket).__setitem__(key, self.__upload(value))
      else:
        super(Hyperparameters, bucket).__setitem__(key, value)
    return bucket
