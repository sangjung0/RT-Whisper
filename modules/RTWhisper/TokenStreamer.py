from RTWhisper.models import Whisper, SileroVad
from RTWhisper.preprocessor import AudioMerger
from RTWhisper.classifier import TokenClassifier
from RTWhisper.composer import Composer
from RTWhisper.data import Param, Context
from RTWhisper.filter import DurationFilter, ProbabilityFilter
from RTWhisper.selector import Selector
from RTWhisper.postprocessor import (
    RecoverTimeoffset,
    AdjustWeightAndOffset,
    SegmentsToTokenWithEOS,
)

from .BaseObject import BaseObject
from .Pipeline import Pipeline
from .Hyperparameters import Hyperparameters


class TokenStreamer(BaseObject):
    def __init__(self, hyperparameters: dict = Hyperparameters()):
        super().__init__()
        self._hyperparameters = hyperparameters
        self.__pipeline: list[Pipeline] = []

        self.__init_pipeline()

    @SileroVad.object
    @Whisper.object
    def __init_pipeline(self, whisper: Whisper, silero_vad: SileroVad):
        self.__pipeline = [
            silero_vad,
            # Use: processed_audio
            # Result: processed_audio, timestamps
            AudioMerger(),
            # Use: processed_audio, prev_processed_audio
            # Result: processed_audio
            whisper,
            # Use: processed_audio, language, prompt
            # Result: tokens, language
            SegmentsToTokenWithEOS(whisper.tokenizer),
            # Use: tokens, language
            # Result: tokens
            RecoverTimeoffset(),
            # Use: tokens, timestamps, prev_audio_sc, prev_timestamps
            # Result: tokens, timestamps, processed_timestamp_conditions
            AdjustWeightAndOffset(
                self._hyperparameters["weighted_and_offset_token_boundary"]
            ),
            # Use: sc_offsete, audio, prev_audio_sc, tokens
            # Result: tokens
            DurationFilter(self._hyperparameters["duration_filter_z"]),
            # Use: tokens, language, statistics
            # Result: tokens, statistics
            ProbabilityFilter(
                self._hyperparameters["probability_filter"]["z"],
                self._hyperparameters["probability_filter"]["min_prob"],
            ),
            # Use: tokens, language, statistics
            # Result: tokens, statistics
            Selector(
                self._hyperparameters["selector"]["search_range_sc"],
                self._hyperparameters["selector"]["threshold"],
                self._hyperparameters["selector"]["padding"],
                self._hyperparameters["selector"]["tolerance"],
            ),
            # Use: tokens, prev_recog
            # Result: tokens
            TokenClassifier(
                self._hyperparameters["classifier_max_prev_sc"],
            ),
            # Use: audio, processed_audio, sc_offset, tokens, processed_timestamp_conditions, prev_audio_sc, timestamps
            # Result: prev_timestamps, prev_audio_sc, sc_offset, completed_words, prev_recog, prev_processed_audio
            Composer(),
            # Use: prev_words, completed_words, language, order
            # Result: completed_dict, prev_words, order
        ]

    def process(self, param: Param):
        context = Context()
        context.bind(param)

        for pipeline in self.__pipeline:
            pipeline.process(context)

        return context.extract()
