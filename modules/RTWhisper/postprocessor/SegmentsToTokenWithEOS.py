from whisper import tokenizer

from RTWhisper import Settings
from RTWhisper import Pipeline

from RTWhisper.data import Context, Token


class SegmentsToTokenWithEOS(Pipeline):
  def __init__(
    self,
    tokenizer:tokenizer.Tokenizer,
    SAMPLE_RATE:int = Settings.MODEL_SAMPLE_RATE,
  ):
    super().__init__()
    self.__tokenizer = tokenizer
    self._SAMPLE_RATE = SAMPLE_RATE

  def process(self, context:Context) -> None:
    segments = context.tokens
    language = context.language

    if not segments: return

    tokens = []
    for segment in segments:
      words = [
        Token(
          int(w.start * self._SAMPLE_RATE), 
          int(w.end * self._SAMPLE_RATE), 
          w.word, language,
          self.__tokenizer.encode(w.word.lower()), w.probability
        ) for w in segment.words
      ]
      start = words[0].start
      end = words[-1].end
      words.append(Token(start, end, segment.text, None, None, 1, False))
      tokens.extend(words)

    context.tokens = tokens
  