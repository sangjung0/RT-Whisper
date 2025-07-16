from logging import Logger


class RTWhisperLogger:
    def __init__(self, logger: Logger, default_level: int = 0, indent: str = "\t"):
        self.logger = logger
        self.__DEFAULT_LEVEL = default_level
        self.__INDENT = indent

    def info(self, message: str, *objects: str, group_level: int = 0, **kwargs):
        message = self.__set_group_indent(message, group_level)
        self.logger.info(message, *objects, **kwargs, stacklevel=2)

    def debug(self, message: str, *objects: str, group_level: int = 0, **kwargs):
        message = self.__set_group_indent(message, group_level)
        self.logger.debug(message, *objects, **kwargs, stacklevel=2)

    def warning(self, message: str, *objects: str, group_level: int = 0, **kwargs):
        message = self.__set_group_indent(message, group_level)
        self.logger.warning(message, *objects, **kwargs, stacklevel=2)

    def error(self, message: str, *objects: str, group_level: int = 0, **kwargs):
        message = self.__set_group_indent(message, group_level)
        self.logger.error(message, *objects, **kwargs, stacklevel=2)

    def critical(self, message: str, *objects: str, group_level: int = 0, **kwargs):
        message = self.__set_group_indent(message, group_level)
        self.logger.critical(message, *objects, **kwargs, stacklevel=2)

    def exception(self, message: str, *objects: str, group_level: int = 0, **kwargs):
        message = self.__set_group_indent(message, group_level)
        self.logger.exception(message, *objects, **kwargs, stacklevel=2)

    def get_child(self, indent_size: int = 1):
        if indent_size < 0:
            raise ValueError("Indent size must be non-negative")
        return RTWhisperLogger(
            logger=self.logger,
            default_level=self.__DEFAULT_LEVEL + indent_size,
            indent=self.__INDENT,
        )

    def __set_group_indent(self, message: str, group_level: int) -> str:
        indent_size = group_level + self.__DEFAULT_LEVEL
        if indent_size <= 0:
            return message
        return f"{self.__INDENT * indent_size}{message}"
