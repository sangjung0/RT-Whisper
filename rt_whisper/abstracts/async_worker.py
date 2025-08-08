from abc import ABC
from typing import Any


class AsyncWorker(ABC):
    def _register_state(self, state: Any) -> None: ...

    # main process
    async def process(self, context: Any) -> None:
        param = self._can_process(context)
        if param is None:
            return
        result = await self._process(param)
        self._update(context, result)

    def _can_process(self, context: Any) -> Any:
        return None

    async def _process(self, param: Any) -> Any: ...
    def _update(self, context: Any, result: Any) -> None: ...

    # post process
    async def _post_process(self, context: Any) -> None:
        param = self._can_post_process(context, result)
        if param is None:
            return
        result = await self._post_process(param)
        self._post_update(context, result)

    def _can_post_process(self, context: Any, result: Any) -> Any:
        return None

    async def _post_process(self, param: Any) -> Any: ...
    def _post_update(self, context: Any, result: Any) -> None: ...

    async def context_build(self, context: Any) -> None:
        param = self._can_build(context)
        if param is None:
            return
        result = await self._context_build(param)
        self._context_update(context, result)

    def _can_build(self, context: Any) -> Any:
        return None

    async def _context_build(self, context: Any) -> None: ...
    def _context_update(self, context: Any, result: Any) -> None: ...


__all__ = ["AsyncWorker"]
