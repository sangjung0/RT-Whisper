class Token:
  def __init__(
    self, 
    start:int,            # start sample length
    end:int,              # end sample length
    text:str,               # text of the word
    lang:str,            # language of the word
    tokens:list[int],       # tokens of the word
    probability:float,      # probability of the word
    is_word:bool = True     # is word or not
  ):
    self.__start = start
    self.__end = end
    self.__text = text
    self.__lang = lang
    self.__tokens = tokens
    self.__is_word = is_word 
    self.__probability = probability

  def __str__(self) -> str:
    return f"{self.__start} {self.__end} {self.__text}"

  @property
  def start(self) -> int:
    return self.__start
  @start.setter
  def start(self, value:int) -> None:
    if not isinstance(value, int):
      raise TypeError("start must be int")
    elif value < 0:
      raise ValueError("start must be greater than 0")
    self.__start = value

  @property
  def end(self) -> int:
    return self.__end
  @end.setter
  def end(self, value:int) -> None:
    if not isinstance(value, int):
      raise TypeError("end must be int")
    elif value < 0:
      raise ValueError("end must be greater than 0")
    self.__end = value

  @property
  def text(self) -> str:
    return self.__text
  @text.setter
  def text(self, value:str) -> None:
    if not isinstance(value, str):
      raise TypeError("text must be str")
    self.__text = value

  @property
  def lang(self) -> str:
    return self.__lang
  @lang.setter
  def lang(self, value:str) -> None:
    if not isinstance(value, str):
      raise TypeError("lang must be str")
    self.__lang = value

  @property
  def tokens(self):
    return self.__tokens
  @tokens.setter
  def tokens(self, value:list[int]) -> None:
    if not isinstance(value, list):
      raise TypeError("tokens must be list")
    elif not all(isinstance(i, int) for i in value):
      raise TypeError("tokens must be list of int")
    self.__tokens = value
  
  @property
  def is_word(self):
    return self.__is_word
  @is_word.setter
  def is_word(self, value:bool) -> None:
    if not isinstance(value, bool):
      raise TypeError("is_word must be bool")
    self.__is_word = value

  @property
  def probability(self):
    return self.__probability
  @probability.setter
  def probability(self, value:float) -> None:
    if not isinstance(value, float):
      raise TypeError("probability must be float")
    elif value < 0:
      raise ValueError("probability must be greater than 0")
    self.__probability = value

  def to_dict(self):
    return {
      "start": self.start,
      "end": self.end,
      "text": self.text,
      "lang": self.lang,
      "tokens": self.tokens,
      "probability": self.probability,
      "is_word": self.is_word
    }

  @classmethod
  def from_dict(cls, data):
    return cls(
      start=data["start"],
      end=data["end"],
      text=data["text"],
      lang=data["lang"],
      tokens=data["tokens"],
      probability=data["probability"],
      is_word=data["is_word"]
    )

  def __json__(self):
    return self.to_dict()

