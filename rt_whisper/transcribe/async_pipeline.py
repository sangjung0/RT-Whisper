import asyncio
from rt_whisper.data import TokenState, Param, Result

from .pipeline import Pipeline


class AsyncPipeline(Pipeline):
    async def process(self, param: Param) -> Result:
        if not self._pipeline:
            raise RuntimeError("Pipeline is not initialized with workers.")

        context = TokenState()
        for worker in self._workers:
            for w in worker:
                w._register_state(context)

        context.bind(param)

        for worker_group in self._pipeline:
            for worker in worker_group[0]:
                fut = worker.process(context)
                if asyncio.iscoroutine(fut):
                    await fut
            for worker in worker_group[1]:
                fut = worker.post_process(context)
                if asyncio.iscoroutine(fut):
                    await fut
            for worker in worker_group[2]:
                fut = worker.context_build(context)
                if asyncio.iscoroutine(fut):
                    await fut

        return context.extract()
