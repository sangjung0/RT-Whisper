from abc import ABC
from functools import lru_cache, wraps
from typing import Any, Callable, Type, TypeVar, cast

from rt_whisper.util.utils import camel_to_snake

T = TypeVar("T", bound="Singleton")


class Singleton(ABC):
    implementation: "Singleton" = None

    def __init__(self) -> None:
        cls = self.__class__
        if cls.implementation is not None:
            raise Exception(
                f"{self.__class__.__name__} is a singleton class. Use ServerObject.get_instance() instead."
            )

    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(cls: Type[T], *args, **kwargs) -> T:
        if cls.implementation is None:
            cls.implementation = cls(*args, **kwargs)
        return cast(T, cls.implementation)

    @classmethod
    def object(cls, func: Any) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            self = cls.get_instance()
            kwargs[camel_to_snake(self.__class__.__name__)] = self
            return func(*args, **kwargs)

        return wrapper
