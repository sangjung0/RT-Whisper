from typing import Any, Union
from data import Context


class Pipeline:
    def can_process(self, context: Context) -> Union[bool, Any]:
        raise NotImplementedError("Pipeline can_process method not implemented")

    def compute_process(self, param: Any) -> Any:
        raise NotImplementedError("Pipeline compute_process method not implemented")

    def apply_process(self, context: Context, result: Any) -> None:
        raise NotImplementedError("Pipeline apply_process method not implemented")

    def process(self, context: Context) -> None:
        param = self.can_process(context)
        if isinstance(param, bool) and not param:
            return
        result = self.compute_process(param)
        self.apply_process(context, result)
