from RTWhisper import Pipeline
from RTWhisper.data import Context

class AdjustOffset(Pipeline):
  def process(self, context:Context):

    if not context.tokens:
      return

    sc_offset = context.sc_offset
    tokens = context.tokens

    for token in tokens:
      token.start += sc_offset
      token.end += sc_offset

    # context.tokens = tokens