import numpy as np

from .classifier_ import Classifier


class SentenceClassifier(Classifier):
    def can_process(self, context):
        return (
            context.sc_offset,
            context.completed,
            context.audio,
            context.audio_sc,
            context.prev_audio,
            context.prev_audio_sc,
            context.merged_processed_audio,
            context.merged_timestamps,
            context.merged_timestamps_mapping,
            context.prev_sentence,
            context.prev_candidate_sentences,
        )

    def compute_process(self, param):
        (
            sc_offset,
            completed,
            audio,
            audio_sc,
            prev_audio,
            prev_audio_sc,
            merged_processed_audio,
            merged_timestamps,
            merged_timestamps_mapping,
            prev_sentence,
            prev_candidate_sentences,
        ) = param

        if len(merged_processed_audio) == 0:
            return (
                sc_offset + prev_audio_sc + audio_sc,
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
                [],
                None,
                prev_candidate_sentences,
            )

        anchor = (
            (completed[-1].tokens[-1].end - sc_offset)
            if completed
            else merged_timestamps[0]["start"]
        )
        prev_audio_start = 0
        for ts in merged_timestamps:
            if ts["start"] <= anchor <= ts["end"]:
                prev_audio_start += anchor - ts["start"]
                break
            prev_audio_start += ts["end"] - ts["start"]

        merged_audio = np.concatenate([prev_audio, audio])

        prev_audio = merged_audio[anchor:]
        prev_processed_audio = merged_processed_audio[prev_audio_start:]
        prev_timestamps = self._get_prev_timestamps(merged_timestamps, anchor)
        prev_timestamps_mapping = self._get_prev_timestamps_mapping(
            merged_timestamps_mapping, prev_audio_start
        )
        sc_offset += anchor
        prev_sentence = completed[-1] if completed else prev_sentence

        return (
            sc_offset,
            prev_audio,
            prev_processed_audio,
            prev_timestamps,
            prev_timestamps_mapping,
            prev_sentence,
            completed,
        )

    def apply_process(self, context, result):
        (
            sc_offset,
            prev_audio,
            prev_processed_audio,
            prev_timestamps,
            prev_timestamps_mapping,
            prev_sentence,
            completed,
        ) = result

        context.sc_offset = sc_offset
        context.prev_audio = prev_audio
        context.prev_processed_audio = prev_processed_audio
        context.prev_timestamps = prev_timestamps
        context.prev_timestamps_mapping = prev_timestamps_mapping
        context.prev_sentence = prev_sentence
        context.completed = completed
