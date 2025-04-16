import statistics

from RTWhisper import Pipeline
from RTWhisper.data import Context
from RTWhisper.util.utils import update_mean_std

class ProbabilityFilter(Pipeline):
  def __init__(
    self,
    Z_THRESH:float,
    MIN_PROB:float,
  ):
    super().__init__()
    self.__Z_THRESH = Z_THRESH
    self.__MIN_PROB = MIN_PROB

  def can_process(self, context:Context) -> bool:
    if context.merged_processed_audio_sc == 0: return False
    if not context.merged_candidate_tokens: return False
    language = context.language
    return (
      context.merged_candidate_tokens, 
      context.language, 
      context.statistics["probability"]["mean"].get(language, None), 
      context.statistics["probability"]["std"].get(language, None), 
      context.statistics["probability"]["count"].get(language, None)
    )

  def compute_process(self, param:tuple):
    tokens, language, prev_mean, prev_std, prev_n = param

    tokens = [t for t in tokens if t.probability > self.__MIN_PROB[language]]
    X = [t.probability for t in tokens if t.is_word]

    if not X: return tokens, prev_mean, prev_std, prev_n

    N = len(X)
    mean = statistics.mean(X)
    std = statistics.stdev(X) if len(X) > 1 else 0.0
    if prev_mean is not None and prev_std is not None and prev_n is not None:
        mean, std = update_mean_std(
          prev_mean, prev_std, prev_n, 
          mean, std, N
        )
        
    new_tokens = []
    for t in tokens:
      if t.is_word and t.probability < mean and mean - t.probability > self.__Z_THRESH[language] * std:
          continue
      new_tokens.append(t)

    n = N + prev_n if prev_n is not None else N
    return new_tokens, mean, std, n

  def apply_process(self, context:Context, result:tuple) -> None:
    new_tokens, mean, std, n = result
    context.merged_candidate_tokens = new_tokens
    context.statistics["probability"]["mean"][context.language] = mean
    context.statistics["probability"]["std"][context.language] = std
    context.statistics["probability"]["count"][context.language] = n