import torch
from pydantic import BaseModel, Field, ConfigDict


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
