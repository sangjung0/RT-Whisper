import numpy as np

from rt_whisper.data import Context

from .classifier import Classifier
from .data import SentenceClassifierParam, SentenceClassifierResult


class SentenceClassifier(Classifier):
    # override
    def _can_process(self, context:Context) -> SentenceClassifierParam:
        return SentenceClassifierParam.from_context(context)

    # override
    def _process(self, param:SentenceClassifierParam) -> SentenceClassifierResult:

        if len(param.merged_processed_audio) == 0:
            return (
                param.sc_offset + param.prev_audio_sc + param.audio_sc,
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
                [],
                None,
                param.prev_candidate_sentences,
            )

        anchor = (
            (param.completed[-1].tokens[-1].end - param.sc_offset)
            if param.completed
            else param.merged_timestamps[0]["start"]
        )
        prev_audio_start = 0
        for ts in param.merged_timestamps:
            if ts["start"] <= anchor <= ts["end"]:
                prev_audio_start += anchor - ts["start"]
                break
            prev_audio_start += ts["end"] - ts["start"]

        merged_audio = np.concatenate([param.prev_audio, param.audio])

        prev_audio = merged_audio[anchor:]
        prev_processed_audio = param.merged_processed_audio[prev_audio_start:]
        prev_timestamps = self._get_prev_timestamps(param.merged_timestamps, anchor)
        prev_timestamps_mapping = self._get_prev_timestamps_mapping(
            param.merged_timestamps_mapping, prev_audio_start
        )
        sc_offset += anchor
        prev_sentence = param.completed[-1] if param.completed else param.prev_sentence

        return SentenceClassifierResult(
            sc_offset=sc_offset,
            prev_audio=prev_audio,
            prev_processed_audio=prev_processed_audio,
            prev_timestamps=prev_timestamps,
            prev_timestamps_mapping=prev_timestamps_mapping,
            prev_sentence=prev_sentence,
        )

    # override
    def _update(self, context:Context, result:SentenceClassifierResult):
        result.update_context(context)
