from __future__ import annotations
from typing import TYPE_CHECKING

import json

from pathlib import Path

from rt_whisper.abstracts import Worker
from rt_whisper.archiver.data import DataLoaderParam, DataLoaderState, DataLoaderResult
from rt_whisper.archiver.service import dict_to_state

if TYPE_CHECKING:
    from rt_whisper import RTWhisperLogger
    from rt_whisper.data import TokenState


class DataLoader(Worker):
    def __init__(
        self,
        saved_path: Path,
        logger: RTWhisperLogger,
    ):
        super().__init__()
        if saved_path.exists() and saved_path.is_file():
            raise ValueError(f"save_path must be a directory, not a file: {saved_path}")
        if not any(saved_path.iterdir()):
            raise ValueError(f"save_path is empty: {saved_path}. Cannot load data.")

        self.saved_path = saved_path
        self.logger = logger

    # override
    def _register_state(self, state: TokenState) -> None:
        state.set_state(DataLoaderState, DataLoaderState())

    # override
    def _can_process(self, state: TokenState) -> DataLoaderParam:
        ds_state: DataLoaderState = state.get_state(DataLoaderState)
        return DataLoaderParam.from_state(state, ds_state)

    # override
    def _process(self, param: DataLoaderParam) -> DataLoaderResult:
        self.logger.debug(f"Loading data from {self.saved_path}", group_level=1)

        name = param.cnt_file_name
        target = self.saved_path.joinpath(f"{name}.json")
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        with target.open("r") as f:
            d = json.load(f)
        dict_to_state(d, param.token_state)
        name += 1
        return DataLoaderResult(cnt_file_name=name)

    # override
    def _update(self, state: TokenState, result: DataLoaderResult):
        dl_state: DataLoaderState = state.get_state(DataLoaderState)
        result.update_state(dl_state)
