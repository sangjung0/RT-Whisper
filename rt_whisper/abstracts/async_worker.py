from typing import Any

from .worker import Worker


class AsyncWorker(Worker):

    # main process
    async def process(self, state: Any) -> None:
        param = self._can_process(state)
        if param is None:
            return
        result = await self._process(param)
        self._update(state, result)

    async def _process(self, param: Any) -> Any: ...

    # post process
    async def post_process(self, state: Any) -> None:
        param = self._can_post_process(state)
        if param is None:
            return
        result = await self._post_process(param)
        self._post_update(state, result)

    async def _post_process(self, param: Any) -> Any: ...

    async def context_build(self, state: Any) -> None:
        param = self._can_build(state)
        if param is None:
            return
        result = await self._context_build(param)
        self._context_update(state, result)

    async def _context_build(self, state: Any) -> None: ...
