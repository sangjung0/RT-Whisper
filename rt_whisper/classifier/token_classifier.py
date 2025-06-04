import numpy as np

from rt_whisper.data import Context

from .classifier import Classifier
from .data import TokenClassifierParam, TokenClassifierResult


class TokenClassifier(Classifier):
    def __init__(
        self,
        max_overlap_size: int,
    ):
        super().__init__()
        self.__MAX_OVERLAP_SIZE = max_overlap_size

    # override
    def _can_process(self, context: Context):
        return TokenClassifierParam.from_context(context)

    # override
    def _process(self, param: TokenClassifierParam) -> TokenClassifierResult:
        if len(param.merged_vad_chunk) == 0:
            return TokenClassifierResult(
                next_offset=param.offset + param.chunk_size,
                recycle_chunk=np.zeros((0,), dtype=np.float32),
                recycle_vad_chunk=np.zeros((0,), dtype=np.float32),
                recycle_vad_timestamps=[],
                recycle_vad_timestamps_mapping=[],
                recycle_candidate_tokens=[],
                completed_tokens=param.candidate_tokens,
            )

        # TODO 현재는 이렇게 모아서 한번에 업데이트 하지만, ablation study를 위해서도, 또 더 나은 가독성을 위해서 분리하는게 맞음.

        recycle_vad_chunk_start = max(
            0, param.merged_vad_chunk_size - self.__MAX_OVERLAP_SIZE
        )
        recycle_chunk_start = 0
        for tm in param.merged_vad_timestamps_mapping:
            if tm["start"] <= recycle_vad_chunk_start <= tm["end"]:
                recycle_chunk_start = tm["offset"] + recycle_vad_chunk_start
                break

        next_offset = param.offset + param.chunk_size
        merged_chunk = np.concatenate([param.prev_chunk, param.chunk])
        recycle_chunk = merged_chunk[recycle_chunk_start:]
        recycle_vad_chunk = param.merged_vad_chunk[recycle_vad_chunk_start:]

        recycle_vad_timestamps = self._get_prev_timestamps(
            param.vad_timestamps + param.prev_vad_timestamps, recycle_chunk_start
        )
        recycle_vad_timestamps_mapping = self._get_prev_timestamps_mapping(
            param.merged_vad_timestamps_mapping, recycle_vad_chunk_start
        )
        completed_tokens = param.prev_completed_tokens + [
            t for t in param.candidate_tokens if t.end < next_offset
        ]
        recycle_candidate_tokens = [
            t for t in param.candidate_tokens if t not in completed_tokens and t.is_word
        ]

        return TokenClassifierResult(
            next_offset=next_offset,
            recycle_chunk=recycle_chunk,
            recycle_vad_chunk=recycle_vad_chunk,
            recycle_vad_timestamps=recycle_vad_timestamps,
            recycle_vad_timestamps_mapping=recycle_vad_timestamps_mapping,
            recycle_candidate_tokens=recycle_candidate_tokens,
            completed_tokens=completed_tokens,
        )

    def _update(self, context: Context, result: TokenClassifierResult) -> None:
        result.update_context(context)
