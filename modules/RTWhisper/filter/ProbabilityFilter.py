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

  def process(self, context:Context) -> None:
    tokens = context.tokens
    language = context.language
    prev_mean = context.statistics["probability"]["mean"].get(language, None)
    prev_std = context.statistics["probability"]["std"].get(language, None)
    prev_n = context.statistics["probability"]["count"].get(language, None)

    if not tokens: return

    tokens = [t for t in tokens if t.probability > self.__MIN_PROB[language]]
    X = [t.probability for t in tokens if t.is_word]
    
    if not X: return

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

    n = len(X) + prev_n if prev_n is not None else len(X)

    context.tokens = new_tokens
    context.statistics["probability"]["mean"][language] = mean
    context.statistics["probability"]["std"][language] = std
    context.statistics["probability"]["count"][language] = n
    