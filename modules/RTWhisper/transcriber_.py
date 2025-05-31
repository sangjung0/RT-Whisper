from RTWhisper.SentenceStreamer import (
    AdjustOffset,
    AudioMerger,
    RecoverTimeoffset,
    SegmentsToTokenWithEOS,
)
from RTWhisper.composer import SimpleComposer
from RTWhisper.data import Param, Context
from RTWhisper.models import Whisper, SileroVad

from .Hyperparameters import Hyperparameters
from .BaseObject import BaseObject
from .Pipeline import Pipeline


class Transcriber(BaseObject):
    def __init__(
        self,
        hyperparameters: dict = Hyperparameters(),
    ):
        super().__init__()
        self._hyperparameters = hyperparameters
        self.__pipeline: list[Pipeline] = []

        self.__init_pipeline()

    @SileroVad.object
    @Whisper.object
    def __init_pipeline(self, whisper: Whisper, silero_vad: SileroVad):
        self.__pipeline = [
            silero_vad,
            AudioMerger(),
            whisper,
            SegmentsToTokenWithEOS(whisper.tokenizer),
            RecoverTimeoffset(),
            AdjustOffset(),
            SimpleComposer(),
        ]

    def process(self, param: Param):
        context = Context()
        context.bind(param)

        for pipeline in self.__pipeline:
            pipeline.process(context)

        return context.extract()
