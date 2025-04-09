from typing import Any
from .Word import Word

class Sentence:
  def __init__(self, lang:list[str], text:str, words:list[Word]):
    self.__lang = lang
    self.__text = text
    self.__words = words

  def __str__(self) -> str:
    return f"{self.__lang} {self.__text}"
  
  @property
  def lang(self) -> list[str]:
    return self.__lang
  @property
  def text(self) -> str:
    return self.__text
  @property
  def words(self) -> list[Word]:
    return self.__words

  def to_dict(self):
    return {
      "lang": self.lang,
      "text": self.text,
      "words": [word.to_dict() for word in self.words]
    }

  @classmethod
  def from_dict(cls, data:dict[str:Any]):
    return cls(
      lang= data["lang"],
      text= data["text"],
      words= [Word.from_dict(word) for word in data["words"]]
    )

  def __json__(self):
    return self.to_dict()