from pydantic import BaseModel

from .token_ import Token


class Sentence(BaseModel):
    order: int
    lang: list[str]
    text: str
    tokens: list[Token]
