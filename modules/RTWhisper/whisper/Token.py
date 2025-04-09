from .Word import Word

class Token(Word):
  def __init__(
    self, 
    start:float,            # start time in seconds
    end:float,              # end time in seconds
    text:str,               # text of the word
    lang:str,            # language of the word
    tokens:list[str],       # tokens of the word
    probability:float,      # probability of the word
    is_word:bool = True     # is word or not
  ):
    super().__init__(start, end, text, lang)
    self.__tokens = tokens
    self.__is_word = is_word 
    self.__probability = probability

  @property
  def tokens(self):
    return self.__tokens
  
  @property
  def is_word(self):
    return self.__is_word

  @property
  def probability(self):
    return self.__probability

  def to_dict(self):
    d = super().to_dict()
    return {
      **d,
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

