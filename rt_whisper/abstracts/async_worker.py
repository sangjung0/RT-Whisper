from abc import abstractmethod
from typing import Union

from .singleton import Singleton
from .data import Context, Param, Result

class AsyncWorker(Singleton):

    # main process
    async def process(self, context: Context) -> None:
        param = self._can_process(context)
        if param is None:
            return
        result = await self._process(param)
        self._update(context, result)

    @abstractmethod
    def _can_process(self, context: Context) -> Union[None, Param] : ...

    @abstractmethod
    async def _process(self, param:Param) -> Result: ...

    @abstractmethod
    def _update(self, context: Context, result: Result) -> None: ...

    # post process
    async def _post_process(self, context: Context) -> None:
        param = self._can_post_process(context, result)
        if param is None:
            return
        result = await self._post_process(param)
        self._post_update(context, result)

    @abstractmethod
    def _can_post_process(self, context: Context, result: Result) -> Union[None, Param]: ...

    @abstractmethod
    async def _post_process(self, param:Param) -> Result: ...

    @abstractmethod
    def _post_update(self, context: Context, result: Result) -> None: ...
