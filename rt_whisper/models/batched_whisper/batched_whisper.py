from __future__ import annotations
from typing import TYPE_CHECKING
import asyncio
from faster_whisper.transcribe import Segment

from sj_utils.decorator_utils import singleton

from .transcriber import Transcriber
from .data import *

if TYPE_CHECKING:
    from typing import Callable


@singleton
class BatchedWhisper:
    def __init__(self):
        self.main_process = Transcriber.remote()
        self.__callback: dict[str, Callable[[Task], tuple[Segment, object]]] = {}
        self.__callback_lock = asyncio.Lock()
        self.__loop_task = asyncio.get_event_loop().create_task(self.__loop())

    async def run(self):
        await self.main_process.run.remote()

    async def __loop(self):
        while True:
            result: Result = await self.main_process.get_completed_task.remote()
            if result == "STOP":
                break

            async with self.__callback_lock:
                if result.uuid not in self.__callback:
                    print(
                        f"Warning: No callback registered for task {result.uuid}. Skipping."
                    )
                else:
                    self.__callback[result.uuid](result)

    async def transcribe(self, audio: np.ndarray):
        task = Task(audio=audio)

        fut = asyncio.Future()
        async with self.__callback_lock:
            self.__callback[task.uuid] = lambda p: fut.set_result(p)

        await self.main_process.register_task.remote(task)
        result: Result = await fut

        async with self.__callback_lock:
            del self.__callback[task.uuid]

        return result.segments, result.info

    async def stop(self):
        await self.main_process.stop.remote()
        await self.main_process.sign_to_completed.remote("STOP")
        await self.__loop_task
        self.__loop_task = None
