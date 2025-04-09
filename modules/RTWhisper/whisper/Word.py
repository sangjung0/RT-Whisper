class Word:
  def __init__(
    self, 
    start:float,             # start time in seconds
    end:float,               # end time in seconds
    text:str,                 # text of the word
    lang:str
  ):
    self.__start = start
    self.__end = end
    self.__text = text
    self.__lang = lang

  def __str__(self) -> str:
    return f"{self.__start} {self.__end} {self.__text}"

  @property
  def start(self) -> float:
    return self.__start

  @property
  def end(self) -> float:
    return self.__end

  @property
  def text(self) -> str:
    return self.__text

  @property
  def lang(self) -> str:
    return self.__lang

  def to_dict(self):
    return {
      "start": self.start,
      "end": self.end,
      "text": self.text,
      "lang": self.lang
    }

  @classmethod
  def from_dict(cls, data):
    return cls(
      start=data["start"],
      end=data["end"],
      text=data["text"],
      lang=data["lang"]
    )
    
  def __json__(self):
    return self.to_dict()