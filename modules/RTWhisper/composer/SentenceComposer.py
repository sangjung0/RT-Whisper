from RTWhisper.data import Context
from RTWhisper import Tokenizer

from .Composer import Composer

class SentenceComposer(Composer):
  def __init__(
    self,
    max_prev_sent:int
  ):
    super().__init__()
    self.__MAX_PREV_SENT = max_prev_sent
  
  def process(self, context:Context):
    tokens = context.tokens
    language = context.language
    order = context.order

    if not tokens: return
    tokenizer = Tokenizer.get_tokenizer(language)
    
    if tokenizer is None:
      completed_dict, prev_recog, order = self._cut_by_eos(
        tokens, order, self.__MAX_PREV_SENT + 1
      )
    else:
      completed_dict, prev_recog, order = self._cut_by_tokenizer(
        tokenizer, tokens, order, self.__MAX_PREV_SENT + 1
      )    

    context.completed = completed_dict
    context.prev_sentence = completed_dict[order - 1] if order != context.order else context.prev_sentence
    context.order = order
    context.prev_recog = prev_recog