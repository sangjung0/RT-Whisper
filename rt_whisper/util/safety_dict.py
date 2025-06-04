from typing import Any


class SafetyDict(dict):
    def __init__(
        self,
        data: dict = {},
        default_value: Any = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__DEFAULT_VALUE = default_value
        if data:
            self.update(self.__upload(data))

    def __getitem__(self, key):
        if key not in self:
            print(f"Key '{key}' not found in SafetyDict. Returning default value.")
        return super().get(key, self.__DEFAULT_VALUE)

    def __setitem__(self, key, value):
        raise NotImplementedError("SafetyDict is read-only")

    def __upload(self, d: dict):
        default = d["default"] if "default" in d else None
        bucket = SafetyDict(default_value = default)
        for key, value in d.items():
            if isinstance(value, dict):
                super(SafetyDict, bucket).__setitem__(key, self.__upload(value))
            else:
                super(SafetyDict, bucket).__setitem__(key, value)
        return bucket
