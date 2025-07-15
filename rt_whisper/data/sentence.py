from pydantic import BaseModel

from .token import Token


class Sentence(BaseModel):
    order: int
    lang: list[str]
    text: str
    tokens: list[Token]

    def __str__(self):
        return f"[{self.order}] {self.text}"
