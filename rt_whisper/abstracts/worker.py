from abc import ABC
from typing import Any


class Worker(ABC):
    def _register_state(self, state: Any) -> None: ...

    # main process
    def process(self, state: Any) -> None:
        param = self._can_process(state)
        if param is None:
            return
        result = self._process(param)
        self._update(state, result)

    def _can_process(self, state: Any) -> Any:
        return None

    def _process(self, param: Any) -> Any: ...
    def _update(self, state: Any, result: Any) -> None: ...

    # post process
    def post_process(self, state: Any) -> None:
        param = self._can_post_process(state)
        if param is None:
            return
        result = self._post_process(param)
        self._post_update(state, result)

    def _can_post_process(self, state: Any) -> Any:
        return None

    def _post_process(self, param: Any) -> Any: ...
    def _post_update(self, state: Any, result: Any) -> None: ...

    def context_build(self, state: Any) -> None:
        param = self._can_build(state)
        if param is None:
            return
        result = self._context_build(param)
        self._context_update(state, result)

    def _can_build(self, state: Any) -> Any:
        return None

    def _context_build(self, state: Any) -> None: ...
    def _context_update(self, state: Any, result: Any) -> None: ...


__all__ = ["Worker"]
