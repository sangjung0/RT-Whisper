import torch

from pydantic import BaseModel, Field, ConfigDict

from sj_ai_utils.torch import tensor_to_base64, base64_to_tensor


class Token(BaseModel):
    start: int
    end: int
    text: str
    lang: str
    embedding: torch.Tensor | None
    probability: float
    is_word: bool = Field(True)

    # NOTE 임시 조치
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self):
        return f"{self.text} [{self.start}-{self.end}]"

    def __hash__(self):
        # NOTE **절대 key로 사용하지 말 것. 이거는 단순히 코사인 유사도의 lru 캐시를 사용하기 위한 임시 조치임**
        return hash(self.text)

    def __eq__(self, other):
        # NOTE **절대 key로 사용하지 말 것. 이거는 단순히 코사인 유사도의 lru 캐시를 사용하기 위한 임시 조치임**
        if isinstance(other, Token):
            return self.text == other.text
        return False

    def to_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "lang": self.lang,
            "embedding": (
                tensor_to_base64(self.embedding) if self.embedding is not None else None
            ),
            "probability": self.probability,
            "is_word": self.is_word,
        }

    @staticmethod
    def from_dict(data: dict) -> "Token":
        return Token(
            start=data["start"],
            end=data["end"],
            text=data["text"],
            lang=data["lang"],
            embedding=(
                base64_to_tensor(data["embedding"]) if data["embedding"] else None
            ),
            probability=data["probability"],
            is_word=data.get("is_word", True),
        )
