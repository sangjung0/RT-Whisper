from whisper import tokenizer

from RTWhisper import Settings
from RTWhisper import Pipeline

from RTWhisper.data import Context, Token


class SegmentsToToken(Pipeline):
  def __init__(
    self,
    tokenizer:tokenizer.Tokenizer,
    SAMPLE_RATE:int = Settings.MODEL_SAMPLE_RATE,
  ):
    super().__init__()
    self.__tokenizer = tokenizer
    self._SAMPLE_RATE = SAMPLE_RATE

  def can_process(self, context:Context) -> bool:
    if not context.merged_candidate_tokens: return False
    return context.merged_candidate_tokens, context.language
  
  def compute_process(self, param:tuple):
    (segments, language) = param

    new_tokens = []
    for segment in segments:
      tokens = [
        Token(
          start = int(w.start * self._SAMPLE_RATE), 
          end = int(w.end * self._SAMPLE_RATE), 
          text = w.word, 
          lang = language,
          tokens = [], # self.__tokenizer.encode(w.word.lower()), 
          probability = w.probability
        ) for w in segment.words
      ]
      new_tokens.extend(tokens)

    return new_tokens

  def apply_process(self, context:Context, result) -> None:
    context.merged_candidate_tokens = result