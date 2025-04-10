from RTWhisper import Pipeline
from RTWhisper.data import Context

class AdjustWeightAndOffset(Pipeline):
  def __init__(
    self,
    BOUNDARY: float,
  ):
    super().__init__()
    self.__BOUNDARY = BOUNDARY

  def __get_weighted_probability(
    self, 
    probabilities:float, 
    start_sc:int, 
    end_sc:int, 
    duration_sc:int, 
    boundary:int
  ) -> float:
    center = (start_sc + end_sc) / 2
    if center < duration_sc - boundary:
      return probabilities
    return probabilities * ((duration_sc - center)/boundary)
  
  def process(self, context:Context):

    if context.processed_audio is None:
      return

    sc_offset = context.sc_offset
    audio_sc = len(context.audio) + context.prev_audio_sc 
    tokens = context.tokens

    for token in tokens:
      start = token.start
      end = token.end
      token.start = start + sc_offset
      token.end = end + sc_offset
      if not token.is_word:
        continue
        
      token.probability = self.__get_weighted_probability(
        token.probability, start, end, audio_sc, self.__BOUNDARY
      )

    # context.tokens = tokens