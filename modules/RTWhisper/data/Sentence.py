from typing import Any

from .Token import Token

class Sentence:
  def __init__(self, lang:list[str], text:str, tokens:list[Token]):
    self.__lang = lang
    self.__text = text
    self.__tokens = tokens

  def __str__(self) -> str:
    return f"{self.__lang} {self.__text}"
  
  @property
  def lang(self) -> list[str]:
    return self.__lang
  @property
  def text(self) -> str:
    return self.__text
  @property
  def tokens(self) -> list[Token]:
    return self.__tokens

  def to_dict(self):
    return {
      "lang": self.lang,
      "text": self.text,
      "tokens": [token.to_dict() for token in self.tokens]
    }

  @classmethod
  def from_dict(cls, data:dict[str:Any]):
    return cls(
      lang= data["lang"],
      text= data["text"],
      words= [Token.from_dict(token) for token in data["tokens"]]
    )

  def __json__(self):
    return self.to_dict()