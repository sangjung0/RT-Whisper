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

  def _cut_by_eos(self, tokens:list[Token], order:int, completed_length:int):
    self._base_logger.warning("Tokenizer not found. Using default EOS.")

    completed = []
    candidate = []
    language = set()

    completed_index = 0
    start_idx = 0
    limit_completed = completed_length
    for idx in range(len(tokens)):
      limit_completed  -= 1
      token = tokens[idx]
      language.add(token.lang)

      if not token.is_word:
        sentence = Sentence(
          order = order,
          lang = list(language),
          text = token.text,
          tokens = tokens[start_idx:idx]
        )
        if limit_completed < 0: candidate.append(sentence)
        else:
          completed.append(sentence)
          completed_index = idx + 1

        language = set()
        start_idx = idx + 1
        order += 1
      
    return (
      completed, 
      candidate, 
      tokens[completed_index:completed_length], 
      order - len(candidate)
    )

  def _cut_by_tokenizer(
    self, 
    tokenizer, 
    tokens:list[Token], 
    order:int, 
    completed_length:int
  ):

    completed_length -= len([t for t in tokens[:completed_length] if not t.is_word])
    tokens = [t for t in tokens if t.is_word]

    text = ""
    for token in tokens: text += token.text

    completed = []
    candidate = []
    completed_idx = 0
    start_idx = 0
    limit_completed = completed_length
    sents = tokenizer.segment(text)
    for sent in sents:
      len_scent = len(sent)
      for idx in range(start_idx, len(tokens)):
        len_scent -= len(tokens[idx].text)
        if len_scent <= 0:
          end_idx = idx + 1
          break
      else:
        end_idx = len(tokens)
      
      limit_completed = limit_completed - end_idx + start_idx
      sent_tokens = tokens[start_idx:end_idx]
      language = list(set([token.lang for token in sent_tokens]))
      sentence = Sentence(
        order = order, 
        lang = language, 
        text = sent, 
        tokens = sent_tokens
      )
      if limit_completed < 0: candidate.append(sentence)
      else: 
        completed.append(sentence)
        completed_idx = end_idx

      start_idx = end_idx
      order += 1

    return completed, candidate, tokens[completed_idx:completed_length], order - len(candidate)

  def can_process(self, context:Context) -> bool:
    if ( len(context.merged_completed_tokens) == 0 and 
          len(context.merged_candidate_tokens) == 0): return False
    return (
      [*context.merged_completed_tokens, *context.merged_candidate_tokens],
      context.language, context.order,
      len(context.merged_completed_tokens)
    )

  def compute_process(self, param):
    (
      completed_tokens, language, order, completed_length
    ) = param

    language = language if language else completed_tokens[0].lang
    tokenizer = Tokenizer.get_tokenizer(language)

    if tokenizer is None:
      completed, candidate, completed_tokens, order = self._cut_by_eos(
        completed_tokens, order, completed_length
      )
    else:
      completed, candidate, completed_tokens, order = self._cut_by_tokenizer(
        tokenizer, completed_tokens, order, completed_length
      )

    return completed, candidate, completed_tokens, order

  def apply_process(self, context:Context, result):
    completed, candidate, completed_tokens, order = result

    context.completed = completed
    context.candidate = candidate
    context.prev_completed_tokens = completed_tokens
    context.order = order