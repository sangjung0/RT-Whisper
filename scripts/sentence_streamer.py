from models import Whisper, SileroVad
from preprocessor import AudioMerger
from data import Param, Context
from postprocessor import (
    RecoverTimeOffset,
    AdjustOffset,
    SegmentsToTokenWithEOS,
)
from composer import SentenceComposer
from classifier import SentenceClassifier
from core import BaseObject, Hyperparameters
from abstracts import Pipeline


class SentenceStreamer(BaseObject):
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
            RecoverTimeOffset(),
            # Use: tokens, timestamps, prev_audio_sc, prev_timestamps
            # Result: tokens, timestamps, processed_timestamp_conditions
            AdjustOffset(),
            # Use: sc_offset, tokens
            # Result: tokens
            SentenceComposer(self._hyperparameters["sentence_max_prev_sentence"]),
            # Use: tokens, language, order
            # Result: completed, prev_sentence, prev_recog
            SentenceClassifier(),
            # Use: audio, processed_audio, sc_offset, completed, prev_audio_sc, timestamps, order
            # Result: prev_timestamps, prev_audio_sc, sc_offset, prev_processed_audio
        ]

    def process(self, param: Param):
        context = Context()
        context.bind(param)

        for pipeline in self.__pipeline:
            pipeline.process(context)

        return context.extract()
