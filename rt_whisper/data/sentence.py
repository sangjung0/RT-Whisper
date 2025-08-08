from __future__ import annotations
from typing import TYPE_CHECKING

from pydantic import BaseModel

from rt_whisper.data.token import Token

if TYPE_CHECKING:
    pass


class Sentence(BaseModel):
    order: int
    lang: list[str]
    text: str
    tokens: list[Token]

    def __str__(self):
        return f"[{self.order}] {self.text}"

    def to_dict(self):
        return {
            "order": self.order,
            "lang": self.lang,
            "text": self.text,
            "tokens": [token.to_dict() for token in self.tokens],
        }

    @staticmethod
    def from_dict(data: dict) -> "Sentence":
        return Sentence(
            order=data["order"],
            lang=data["lang"],
            text=data["text"],
            tokens=[Token.from_dict(token) for token in data["tokens"]],
        )


__all__ = ["Sentence"]
