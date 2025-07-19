from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from rt_whisper.data import TokenState


@dataclass(slots=True)
class DataSaverState:
    cnt_file_order: int = 0

    def extract(self) -> DataSaverState:
        return DataSaverState(cnt_file_order=self.cnt_file_order)

    def update(self, state: DataSaverState) -> None:
        self.cnt_file_order = state.cnt_file_order


@dataclass(slots=True)
class DataSaverParam:
    token_state: TokenState
    cnt_file_name: int

    @staticmethod
    def from_state(state: TokenState, ds_state: DataSaverState) -> DataSaverParam:
        return DataSaverParam(token_state=state, cnt_file_name=ds_state.cnt_file_order)


@dataclass(slots=True)
class DataSaverResult:
    cnt_file_name: int

    def update_state(self, state: DataSaverState):
        state.cnt_file_order = self.cnt_file_name


@dataclass(slots=True)
class DataLoaderState:
    cnt_file_order: int = 0

    def extract(self) -> DataLoaderState:
        return DataLoaderState(cnt_file_order=self.cnt_file_order)

    def update(self, state: DataLoaderState) -> None:
        self.cnt_file_order = state.cnt_file_order


@dataclass(slots=True)
class DataLoaderParam:
    token_state: TokenState
    cnt_file_name: int

    @staticmethod
    def from_state(state: TokenState, dl_state: DataLoaderState) -> DataLoaderParam:
        return DataLoaderParam(token_state=state, cnt_file_name=dl_state.cnt_file_order)


@dataclass(slots=True)
class DataLoaderResult:
    cnt_file_name: int

    def update_state(self, state: DataLoaderState):
        state.cnt_file_order = self.cnt_file_name


__all__ = [
    "DataSaverParam",
    "DataLoaderParam",
    "DataSaverResult",
    "DataLoaderResult",
    "DataSaverState",
    "DataLoaderState",
]
