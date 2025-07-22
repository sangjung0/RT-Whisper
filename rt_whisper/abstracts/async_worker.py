from abc import abstractmethod
from typing import Union, Any

from sj_utils.decorator import singleton


@singleton
class AsyncWorker:

    # main process
    async def process(self, context: Any) -> None:
        param = self._can_process(context)
        if param is None:
            return
        result = await self._process(param)
        self._update(context, result)

    @abstractmethod
    def _can_process(self, context: Any) -> Union[None, Any]: ...

    @abstractmethod
    async def _process(self, param: Any) -> Any: ...

    @abstractmethod
    def _update(self, context: Any, result: Any) -> None: ...

    # post process
    async def _post_process(self, context: Any) -> None:
        param = self._can_post_process(context, result)
        if param is None:
            return
        result = await self._post_process(param)
        self._post_update(context, result)

    @abstractmethod
    def _can_post_process(
        self, context: Any, result: Any
    ) -> Union[None, Any]: ...

    @abstractmethod
    async def _post_process(self, param: Any) -> Any: ...

    @abstractmethod
    def _post_update(self, context: Any, result: Any) -> None: ...
