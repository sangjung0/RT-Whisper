from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
import uuid

if TYPE_CHECKING:
    import numpy as np
    from faster_whisper.transcribe import Segment


@dataclass
class Task:
    audio: np.ndarray
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Result:
    uuid: str
    segments: list[Segment]
    info: object
