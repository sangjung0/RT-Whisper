import numpy as np

from RTWhisper.data import Context
from .classifier_ import Classifier


class TokenClassifier(Classifier):
    def __init__(
        self,
        MAX_PREV_SC: int,
    ):
        super().__init__()
        self.__MAX_PREV_SC = MAX_PREV_SC

    def can_process(self, context: Context) -> bool:
        return (
            context.sc_offset,
            context.audio,
            context.audio_sc,
            context.prev_audio,
            context.prev_audio_sc,
            context.prev_completed_tokens,
            context.merged_processed_audio,
            context.merged_processed_audio_sc,
            context.merged_timestamps,
            context.merged_timestamps_mapping,
            context.merged_candidate_tokens,
        )

    def compute_process(self, param):
        (
            sc_offset,
            audio,
            audio_sc,
            prev_audio,
            prev_audio_sc,
            prev_completed_tokens,
            merged_processed_audio,
            merged_processed_audio_sc,
            merged_timestamps,
            merged_timestamps_mapping,
            merged_candidate_tokens,
        ) = param

        if len(merged_processed_audio) == 0:
            return (
                sc_offset + prev_audio_sc + audio_sc,  # sc_offset
                np.zeros((0,), dtype=np.float32),  # prev_audio
                np.zeros((0,), dtype=np.float32),  # prev_processed_audio
                [],  # prev_timestamps
                [],  # prev_timestamps_mapping
                [],  # prev_candidate_tokens
                merged_candidate_tokens,  # merged_completed_tokens
            )

        prev_audio_start = max(
            0, merged_processed_audio_sc - self.__MAX_PREV_SC["default"]
        )
        anchor = 0
        for tm in merged_timestamps_mapping:
            if tm["start"] <= prev_audio_start <= tm["end"]:
                anchor = tm["offset"] + prev_audio_start
                break

        merged_audio = np.concatenate([audio, prev_audio])

        prev_audio = merged_audio[anchor:]
        prev_processed_audio = merged_processed_audio[prev_audio_start:]
        prev_timestamps = self._get_prev_timestamps(merged_timestamps, anchor)
        prev_timestamps_mapping = self._get_prev_timestamps_mapping(
            merged_timestamps_mapping, prev_audio_start
        )
        sc_offset += anchor
        merged_completed_tokens = prev_completed_tokens + [
            t for t in merged_candidate_tokens if t.end < sc_offset
        ]
        prev_candidate_tokens = [
            t
            for t in merged_candidate_tokens
            if t not in merged_completed_tokens and t.is_word
        ]

        return (
            sc_offset,
            prev_audio,
            prev_processed_audio,
            prev_timestamps,
            prev_timestamps_mapping,
            prev_candidate_tokens,
            merged_completed_tokens,
        )

    def apply_process(self, context: Context, result: tuple):
        (
            sc_offset,
            prev_audio,
            prev_processed_audio,
            prev_timestamps,
            prev_timestamps_mapping,
            prev_candidate_tokens,
            merged_completed_tokens,
        ) = result
        context.sc_offset = sc_offset
        context.prev_audio = prev_audio
        context.prev_processed_audio = prev_processed_audio
        context.prev_timestamps = prev_timestamps
        context.prev_timestamps_mapping = prev_timestamps_mapping
        context.prev_candidate_tokens = context.merged_candidate_tokens = (
            prev_candidate_tokens
        )
        context.merged_completed_tokens = merged_completed_tokens
