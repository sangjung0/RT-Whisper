from RTWhisper.data import Context
from RTWhisper import Tokenizer

from .Composer import Composer

class SimpleComposer(Composer):
  def __init__(self):
    super().__init__()
  
  def process(self, context:Context):
    tokens = context.tokens
    language = context.language
    order = context.order

    if not tokens: return
    tokenizer = Tokenizer.get_tokenizer(language)
    
    if tokenizer is None:
      completed_dict, prev_recog, order = self._cut_by_eos(
        tokens, order, 0
      )
    else:
      completed_dict, prev_recog, order = self._cut_by_tokenizer(
        tokenizer, tokens, order, 0
      )    

    context.completed = completed_dict
    context.order = order
    context.prev_recog = prev_recog