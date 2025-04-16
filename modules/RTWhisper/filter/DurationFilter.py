import statistics

from RTWhisper import Pipeline
from RTWhisper.data import Context
from RTWhisper.util.utils import update_mean_std

class DurationFilter(Pipeline):
  def __init__(
    self,
    Z_THRESH:dict[str:float]
  ):
    super().__init__()
    self.__Z_THRESH = Z_THRESH

  def can_process(self, context:Context) -> bool:
    if context.merged_processed_audio_sc == 0: return False
    if not context.merged_candidate_tokens: return False
    language = context.language
    return (
      context.merged_candidate_tokens, 
      context.language, 
      context.statistics["duration"]["mean"].get(language, None), 
      context.statistics["duration"]["std"].get(language, None), 
      context.statistics["duration"]["count"].get(language, None)
    )

  def compute_process(self, param:tuple) -> tuple:
    tokens, language, prev_mean, prev_std, prev_n = param

    X = [
      (t.end - t.start)/len(t.text.strip()) 
      for t in tokens if t.is_word
    ]

    if not X: return tokens, prev_mean, prev_std, prev_n

    adjusted_X = [x for x in X if x > 0]
    N = len(adjusted_X)
    mean = statistics.mean(adjusted_X)
    std = statistics.stdev(adjusted_X) if len(adjusted_X) > 1 else 0.0
    if prev_mean is not None and prev_std is not None and prev_n is not None:
        mean, std = update_mean_std(
          prev_mean, prev_std, prev_n, 
          mean, std, N
        )
        
    new_tokens = []
    X_iter = iter(X)
    for t in tokens:
      if t.is_word:
        x = next(X_iter)
        if x < mean and mean - x > self.__Z_THRESH[language] * std:
          continue
      new_tokens.append(t)
    
    n = N + prev_n if prev_n is not None else N
    return new_tokens, mean, std, n

  def apply_process(self, context:Context, result:tuple) -> None:
    new_tokens, mean, std, n = result
    language = context.language

    context.merged_candidate_tokens = new_tokens
    context.statistics["duration"]["mean"][language] = mean
    context.statistics["duration"]["std"][language] = std
    context.statistics["duration"]["count"][language] = n