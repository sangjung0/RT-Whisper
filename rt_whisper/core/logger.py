import logging
from pathlib import Path
from typing import Any, Callable
from rich.logging import RichHandler

from .state import config

LOG_DIR_PATH = Path(config.rt_whisper.log_dir)
LOG_DIR_PATH.mkdir(exist_ok=True)
DEFAULT_LOG_LEVEL = config.rt_whisper.log_level

def generate(name: str, level: int = DEFAULT_LOG_LEVEL):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    logger.propagate = False

    logger.addHandler(RichHandler())

    log_path = LOG_DIR_PATH / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger

logger = generate("core", DEFAULT_LOG_LEVEL)

def logger_wrap(func: Any) -> Callable[..., Any]:
    def wrapper(*args, **kwargs) -> Any:
        kwargs["logger"] = logger
        return func(*args, **kwargs)
    return wrapper
