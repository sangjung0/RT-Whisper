import numpy as np

from RTWhisper import Pipeline
from RTWhisper.data import Context


class AudioMerger(Pipeline):

    def can_process(self, context):
        if context.processed_audio_sc == 0:
            return False
        return (
            context.processed_audio,
            context.prev_processed_audio,
            context.prev_processed_audio_sc,
        )

    def compute_process(self, param):
        audio, prev_audio, prev_sc = param
        if prev_sc == 0:
            return audio
        return np.concatenate([prev_audio, audio], axis=0)

    def apply_process(self, context, result):
        context.merged_processed_audio = result
