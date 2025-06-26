from __future__ import annotations
from typing import TYPE_CHECKING
import asyncio
from faster_whisper import WhisperModel
import ray
import traceback

from rt_whisper.core.state import config
from .data import Result
from .service import *

if TYPE_CHECKING:
    from typing import Any


@ray.remote(num_cpus=1, num_gpus=1)
class Transcriber:
    def __init__(
        self,
        model_size: str = config.rt_whisper.model_size,
        device: str = config.rt_whisper.model_device,
        compute_type: str = config.rt_whisper.model_compute_type,
        batch_size: int = config.rt_whisper.model_batch_size,
        beam_size: int = config.rt_whisper.model_beam_size,
    ):
        # NOTE 이거 나중에 함수형으로 해서 여기서 초기화하는 방향으로 해보고 안되면 바꾸자
        self.__task_queue = asyncio.Queue()
        self.__completed_queue = asyncio.Queue()
        self.__run_task: asyncio.Future | None = None
        self.__BEAM_SIZE = beam_size
        self.__BATCH_SIZE = batch_size

        self.__model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    async def register_task(self, task) -> None:
        self.__task_queue.put_nowait(task)

    async def get_completed_task(self) -> Result:
        return await self.__completed_queue.get()

    def run(self):
        if self.__run_task is not None:
            raise RuntimeError("MainProcess is already running.")
        self.__run_task = asyncio.get_running_loop().create_task(self.__run())

    async def __run(self):
        while True:
            print("Waiting for tasks...")
            batch = await collect_batch(self.__task_queue, self.__BATCH_SIZE)
            print(f"Received batch: {batch}")

            tasks, not_task = separate_tasks(batch)
            print(f"Tasks: {tasks}, Not tasks: {not_task}")

            try:
                if tasks:
                    results: list[Result] = transcribe(
                        self.__model,
                        tasks,
                        batch_size=len(batch),
                        beam_size=self.__BEAM_SIZE,
                    )
                    print(f"Transcription results: {results}")

                    for result in results:
                        self.__completed_queue.put_nowait(result)
                        print(f"Result added to completed queue: {result}")
            except Exception as e:
                traceback.print_exc()

            for task in not_task:
                if task == "STOP":
                    return

    async def stop(self):
        if self.__run_task is None:
            raise RuntimeError("MainProcess is not running.")
        await self.__task_queue.put("STOP")
        await self.__run_task
        self.__run_task = None

    async def sign_to_completed(self, sign: Any):
        await self.__completed_queue.put(sign)
