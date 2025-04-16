from RTWhisper import Pipeline
from RTWhisper.data import Context

class AdjustOffset(Pipeline):

  def can_process(self, context:Context):
    if not context.merged_candidate_tokens:
      return False
    return context.merged_candidate_tokens, context.sc_offset

  def compute_process(self, param):
    (tokens, sc_offset) = param

    for token in tokens:
      token.start += sc_offset
      token.end += sc_offset

    # return tokens

  def apply_process(self, context:Context, result):
    # context.merged_candidate_tokens = result
    pass