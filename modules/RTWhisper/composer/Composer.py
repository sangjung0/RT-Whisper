from logging import Logger

from RTWhisper.data import Sentence, Context, Token
from RTWhisper import BaseLogger, Pipeline, Tokenizer

class Composer(Pipeline):

  @BaseLogger.object
  def __init__(
    self,
    base_logger:Logger
  ):
    super().__init__()
    self._base_logger = base_logger

  def _cut_by_eos(self, tokens:list[Token], order:int, tail:int):
    self._base_logger.warning("Tokenizer not found. Using default EOS.")

    start_idx = len(tokens) - 1
    tail += 1
    while tail > 0 and start_idx > 0:
      if tokens[start_idx].is_word: tail -= 1
    tokens = tokens[:start_idx + 1]

    completed_dict = {}
    language = set()
    start_idx = 0
    for idx in range(len(tokens)):
      token = tokens[idx]
      language.add(token.lang)
      if not token.is_word:
        completed_dict[order] = Sentence(
          list(language), 
          token.text,
          tokens[start_idx:idx]
        )
        start_idx = idx + 1
        order += 1
      
    return completed_dict, tokens[start_idx:], order

  def _cut_by_tokenizer(
    self, 
    tokenizer, 
    tokens:list[Token], 
    order:int, 
    tail:int
  ):

    tokens = [t for t in tokens if t.is_word]
    text = ""
    for token in tokens:
      text += token.text

    completed_dict = {}
    start_idx = 0
    sents = tokenizer.segment(text)
    for sent in (sents[:-tail] if tail > 0 else sents):
      len_scent = len(sent)
      for idx in range(start_idx, len(tokens)):
        len_scent -= len(tokens[idx].text)
        if len_scent <= 0:
          end_idx = idx + 1
          break
      else:
        end_idx = len(tokens)

      language = list(set(
        w.lang for w in tokens[start_idx:end_idx]
      ))

      completed_dict[order] = Sentence(
        language,
        sent,
        tokens[start_idx:end_idx]
      )
      start_idx = end_idx
      order += 1

    return completed_dict, tokens[start_idx:], order

  def process(self, context:Context):
    completed_words = [*context.prev_words, *context.completed_words]
    language = context.language
    order = context.order

    if not completed_words: return
    if not language:
      language = completed_words[0].lang
    tokenizer = Tokenizer.get_tokenizer(language)
    
    if tokenizer is None:
      completed_dict, completed_words, order = self._cut_by_eos(
        completed_words, order, 1
      )
    else:
      completed_dict, completed_words, order = self._cut_by_tokenizer(
        tokenizer, completed_words, order, 1
      )

    context.completed = completed_dict
    context.prev_words = completed_words
    context.order = order