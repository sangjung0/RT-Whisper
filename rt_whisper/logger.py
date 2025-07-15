from logging import Logger


class RTWhisperLogger:
    def __init__(self, logger: Logger, indent: str = "\t"):
        self.logger = logger
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

    def __set_group_indent(self, message: str, group_level: int) -> str:
        if group_level <= 0:
            return message
        return f"{self.__INDENT * group_level}{message}"
