from typing import Any
import asyncio
from faster_whisper import WhisperModel

from .data import *


async def collect_batch(queue: asyncio.Queue, batch_size: int) -> list[Task | Any]:
    batch = []
    while len(batch) < batch_size:
        if queue.empty():
            if len(batch) != 0:
                break
            task = await queue.get()
        else:
            task = queue.get_nowait()
        batch.append(task)

    return batch


def separate_tasks(tasks: list[Task | Any]):
    task_list = [t for t in tasks if isinstance(t, Task)]
    not_task_list = [t for t in tasks if t not in task_list]
    return task_list, not_task_list


def transcribe(model: WhisperModel, tasks: list[Task], batch_size: int, beam_size: int):
    segments, infos = model.transcribe(
        [t.audio for t in tasks],
        batch_size=batch_size,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=False,
    )

    return [
        Result(uuid=t.uuid, segments=seg, info=info)
        for t, seg, info in zip(tasks, segments, infos)
    ]
