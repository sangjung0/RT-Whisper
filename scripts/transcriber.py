from postprocessor import AdjustOffset, RecoverTimeOffset, SegmentsToTokenWithEOS
from preprocessor import AudioMerger
from composer import SimpleComposer
from data import Param, Context
from models import Whisper, SileroVad
from core import Hyperparameters, BaseObject
from abstracts import Pipeline


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
            RecoverTimeOffset(),
            AdjustOffset(),
            SimpleComposer(),
        ]

    def process(self, param: Param):
        context = Context()
        context.bind(param)

        for pipeline in self.__pipeline:
            pipeline.process(context)

        return context.extract()
