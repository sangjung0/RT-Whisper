from __future__ import annotations
from typing import TYPE_CHECKING

import time
import json

from pathlib import Path

from rt_whisper.abstracts import Worker
from rt_whisper.archiver.data import DataSaverParam, DataSaverState, DataSaverResult
from rt_whisper.archiver.service import state_to_dict

if TYPE_CHECKING:
    from rt_whisper import RTWhisperLogger
    from rt_whisper.data import TokenState


class DataSaver(Worker):
    def __init__(
        self,
        save_path: Path,
        logger: RTWhisperLogger,
    ):
        super().__init__()
        if save_path.exists() and save_path.is_file():
            raise ValueError(f"save_path must be a directory, not a file: {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        if any(save_path.iterdir()):
            logger.warning(
                f"save_path is not empty: {save_path}. Existing files will be overwritten. 3s after process start."
            )
            time.sleep(3)

        self.save_path = save_path
        self.logger = logger

    # override
    def _register_state(self, state: TokenState) -> None:
        state.set_state(DataSaverState, DataSaverState())

    # override
    def _can_process(self, state: TokenState) -> DataSaverParam:
        ds_state: DataSaverState = state.get_state(DataSaverState)
        return DataSaverParam.from_state(state, ds_state)

    # override
    def _process(self, param: DataSaverParam) -> DataSaverResult:
        self.logger.debug(f"Saving data to {self.save_path}", group_level=1)

        name = param.cnt_file_name
        d = state_to_dict(param.token_state)
        target = self.save_path.joinpath(f"{name}.json")
        if target.exists():
            self.logger.warning(f"File already exists: {target}. Overwriting.")
        with target.open("w") as f:
            json.dump(d, f, indent=4)

        name += 1
        return DataSaverResult(cnt_file_name=name)

    # override
    def _update(self, state: TokenState, result: DataSaverResult):
        ds_state: DataSaverState = state.get_state(DataSaverState)
        result.update_state(ds_state)


__all__ = ["DataSaver"]
