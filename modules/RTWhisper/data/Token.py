from pydantic import BaseModel, Field

class Token(BaseModel):
  start:int
  end:int
  text:str
  lang:str
  tokens:list[int]
  probability:float
  is_word:bool = Field(True)