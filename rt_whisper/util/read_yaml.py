from pathlib import Path
from types import SimpleNamespace
import yaml
from dotenv import load_dotenv
import os
import re

load_dotenv()

PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def replacer(match):
    return os.getenv(match.group(1), match.group(2))


def replace(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            replace(v)
        elif isinstance(v, list):
            for idx in range(len(v)):
                if isinstance(v[idx], dict):
                    replace(v[idx])
                elif isinstance(v[idx], str):
                    v[idx] = PATTERN.sub(replacer, v[idx])
        elif isinstance(v, str):
            d[k] = PATTERN.sub(replacer, v)


def to_namespace(d: dict):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [to_namespace(i) for i in d]
    else:
        return d


class ReadYaml:
    def __init__(self, path: Path) -> None:
        super().__init__()

        self.path = path
        self.dict = self.__load_yaml()
        self.namespace = to_namespace(self.dict)

    def __load_yaml(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        replace(config)
        return config
