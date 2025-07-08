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
