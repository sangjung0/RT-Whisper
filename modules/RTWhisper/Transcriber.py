from RTWhisper.SentenceStreamer import AdjustOffset, AudioMerger, RecoverTimeoffset, SegmentsToTokenWithEOS
from RTWhisper.composer import SimpleComposer
from RTWhisper.data import Param, Context
from RTWhisper.models import Whisper, SileroVad
from RTWhisper import Hyperparameters, BaseObject, Pipeline, Settings


class Transcriber(BaseObject):
  def __init__(
    self,
    hyperparameters:dict = Hyperparameters(),
  ):
    super().__init__()
    self._hyperparameters = hyperparameters
    self.__pipeline:list[Pipeline] = []

    self.__init_pipeline()

  @Whisper.object
  def __init_pipeline(self, whisper:Whisper):
    self.__pipeline = [
      SileroVad(Settings.MODEL_SAMPLE_RATE),
      AudioMerger(),
      whisper,
      SegmentsToTokenWithEOS(whisper.tokenizer),
      RecoverTimeoffset(),
      AdjustOffset(),
      SimpleComposer()
    ]

  def process(self, param:Param):
    context = Context()
    context.bind(param)
    
    for pipeline in self.__pipeline:
      pipeline.process(context)
      
    return context.extract()