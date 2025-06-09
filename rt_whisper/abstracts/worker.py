from abc import ABC
from .data import Context, Param, Result


class Worker(ABC):

    # main process
    def process(self, context: Context) -> None:
        param = self._can_process(context)
        if param is None:
            return
        result = self._process(param)
        self._update(context, result)

    def _can_process(self, context: Context) -> Param:
        return None

    def _process(self, param: Param) -> Result: ...

    def _update(self, context: Context, result: Result) -> None: ...

    # post process
    def post_process(self, context: Context) -> None:
        param = self._can_post_process(context)
        if param is None:
            return
        result = self._post_process(param)
        self._post_update(context, result)

    def _can_post_process(self, context: Context) -> Param:
        return None

    def _post_process(self, param: Param) -> Result: ...

    def _post_update(self, context: Context, result: Result) -> None: ...

    def recycle(self, context: Context) -> None:
        param = self._need_recycle(context)
        if param is None:
            return
        result = self._recycle(param)
        self._recycle_update(context, result)

    def _need_recycle(self, context: Context) -> Param: ...

    def _recycle(self, context: Param) -> None: ...

    def _recycle_update(self, context: Context, result: Result) -> None: ...
