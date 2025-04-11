from RTWhisper.models import Whisper, SileroVad
from RTWhisper.preprocessor import AudioMerger
from RTWhisper import Settings
from RTWhisper.data import Param, Context
from RTWhisper.postprocessor import RecoverTimeoffset, AdjustOffset, SegmentsToTokenWithEOS
from RTWhisper.composer import SentenceComposer
from RTWhisper.classifier import SentenceClassifier

from .BaseObject import BaseObject
from .Pipeline import Pipeline
from .Hyperparameters import Hyperparameters

class SentenceStreamer(BaseObject):
  def __init__(
    self,
    hyperparameters:dict = Hyperparameters()
  ):
    super().__init__()
    self._hyperparameters = hyperparameters
    self.__pipeline:list[Pipeline] = []

    self.__init_pipeline()

  @Whisper.object
  def __init_pipeline(self, whisper:Whisper):
    self.__pipeline = [
      SileroVad(Settings.MODEL_SAMPLE_RATE),
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
      AdjustOffset(),
      # Use: sc_offset, tokens
      # Result: tokens
      SentenceComposer(self._hyperparameters["sentence_max_prev_sentence"]),
      # Use: tokens, language, order
      # Result: completed, prev_sentence, prev_recog
      SentenceClassifier()
      # Use: audio, processed_audio, sc_offset, completed, prev_audio_sc, timestamps, order
      # Result: prev_timestamps, prev_audio_sc, sc_offset, prev_processed_audio
    ]

  def process(self, param:Param):
    context = Context()
    context.bind(param)

    for pipeline in self.__pipeline:
      pipeline.process(context)
    
    return context.extract()
    