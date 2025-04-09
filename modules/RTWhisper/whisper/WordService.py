from logging import Logger
import re
import statistics
from typing import Optional
import numpy as np
from rapidfuzz.distance import Levenshtein

from RTWhisper import Settings
from RTWhisper import BaseLogger 
from RTWhisper.util.utils import update_mean_std

from .Params import WordParams, Hyperparameters, WordReturn
from .Service import Service
from .Token import Token
from .Sentence import Sentence
from .Tokenizer import Tokenizer
from .Params import WordParams


class WordService(Service):
  def __init__(
    self,
    max_prev_time:float = Settings.MODEL_WORD_MAX_PREV_TIME,
    hyperparameters:dict = Hyperparameters()
  ):
    super().__init__()
    self.__MAX_PREV_TIME = max_prev_time
    self.__HYPERPARAMETERS = hyperparameters
  
  @BaseLogger.object
  def transcribe(
    self,
    wp:WordParams,
    base_logger:Logger = None,
  ):

    # concatenate the previous audio and the current audio
    audio = (wp.audio if wp.prev_audio is None 
             else np.concatenate([wp.prev_audio, wp.audio]))

    # translate the audio to text
    # if language is None, use model's language
    segments, info = self._whisper.transcribe(audio, wp.language)
    
    audio_duration = len(audio) / self._SAMPLE_RATE
    language = info.language 
    # segment word and adjust the time offset and probability
    tokens = self.__segment_word(wp.time_offset, audio_duration, segments, language)

    # filter the tokens by duration. duration - mean / std < z_thresh
    (tokens, 
     new_dura_mean, 
     new_dura_std, 
     new_dura_count) = self.__filter_by_duration_per_char(
      tokens, wp.prev_dura_mean[language], 
      wp.prev_dura_std[language], wp.prev_dura_count[language],
      self.__HYPERPARAMETERS["filter_by_duration_z"][language]
    ) 

    # filter the tokens by probability. probability - mean / std < z_thresh
    (tokens, 
     new_prob_mean, 
     new_prob_std, 
     new_prob_count) = self.__filter_by_probability(
      tokens, wp.prev_prob_mean[language], 
      wp.prev_prob_std[language], wp.prev_prob_count[language],
      self.__HYPERPARAMETERS["filter_by_probability"]["z"][language],
      self.__HYPERPARAMETERS["filter_by_probability"]["min_prob"][language]
    )

    # combine the previous recognized words and the current tokens
    tokens = self.__combine(
      wp.prev_recog, tokens,
      self.__HYPERPARAMETERS["combine"]["search_range_time"][language],
      self.__HYPERPARAMETERS["combine"]["threshold"][language],
      self.__HYPERPARAMETERS["combine"]["tolerance"][language]
    )

    # make new metadata
    (completed_words, 
     prev_recog, 
     prev_audio, 
     new_time_offset) = self.__refine(
       audio, wp.time_offset, tokens,
        self.__HYPERPARAMETERS["refine_tolerance"][language]
      )

    # segment the tokens by scentence
    (completed_dict, 
     prev_words, 
     order) = self.__segment_scent(
       [*wp.prev_words, *completed_words], language, wp.order
      )

    # update statistics
    result = WordReturn(
      completed_dict=completed_dict,
      order=order,
      time_offset=new_time_offset,
      prev_audio=prev_audio,
      prev_words=prev_words,
      prev_recog=prev_recog,
      prev_prob_mean={**wp.prev_prob_mean, language: new_prob_mean},
      prev_prob_std={**wp.prev_prob_std, language: new_prob_std},
      prev_prob_count={**wp.prev_prob_count, language: new_prob_count},
      prev_dura_mean={**wp.prev_dura_mean, language: new_dura_mean},
      prev_dura_std={**wp.prev_dura_std, language: new_dura_std},
      prev_dura_count={**wp.prev_dura_count, language: new_dura_count},
    )

    return result, audio 

  def _get_weighted_probability(
    self, 
    probabilities:float, 
    start:float, 
    end:float, 
    duration:float, 
    boundary:float
  ) -> float:
    center = (start + end) / 2
    if center > boundary:
      if center < duration - boundary:
        return probabilities
      return probabilities * ((duration - center)/boundary) ** 2
    return probabilities * (center/boundary) ** 2

  def __segment_word(
    self, 
    time_offset:float,
    audio_duration:float,
    segments:list, 
    language:str
  ) -> list[Token]:
    result = []
    for segment in segments:
      words = [Token(
        w.start + time_offset, 
        w.end + time_offset, 
        w.word, 
        language,
        self._whisper.tokenizer.encode(w.word.lower()),
        self._get_weighted_probability(
          w.probability, w.start, w.end, 
          audio_duration, 
          self.__HYPERPARAMETERS["weighted_prob_boundary"]
        ),
      ) for w in segment.words]
      start = words[0].start
      end = words[-1].end
      words.append(Token(start, end, "<EOS/>", None, None, 1, False))
      result.extend(words)

    return result
  
  def __filter_by_duration_per_char(
    self,
    tokens: list[Token],
    prev_mean: Optional[float] = None,
    prev_std: Optional[float] = None,
    prev_n: Optional[int] = None,
    z_thresh: float = 2.0
  ) -> tuple[list[Token], float, float, int]:
    X = [
      (t.end - t.start)/len(t.text.strip()) 
      for t in tokens if t.is_word
    ]
    
    if not X:
        return tokens, prev_mean, prev_std, 0

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
        if x < mean and mean - x > z_thresh * std:
          continue
      new_tokens.append(t)

    n = N + prev_n if prev_n is not None else N
    return new_tokens, mean, std, n

  def __filter_by_probability(
    self,
    tokens: list[Token],
    prev_mean: Optional[float] = None,
    prev_std: Optional[float] = None,
    prev_n: Optional[int] = None,
    z_thresh: float = 1.0,
    min_prob: float = 0.4
  ) -> tuple[list[Token], float, float, int]:
    tokens = [t for t in tokens if t.probability > min_prob]
    X = [t.probability for t in tokens if t.is_word]
    
    if not X:
        return tokens, prev_mean, prev_std, 0

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
        if t.is_word and t.probability < mean and mean - t.probability > z_thresh * std:
            continue
        new_tokens.append(t)

    n = len(X) + prev_n if prev_n is not None else len(X)
    return new_tokens, mean, std, n

  def __token_iou(
    self, 
    A:Token, 
    B:Token, 
    padding:float = 0.2, 
    smooth:float = 1e-6
  ) -> float:
    a1 = max(0, A.start - padding)
    b1 = A.end + padding
    a2 = max(0, B.start - padding)
    b2 = B.end + padding

    inner = max(0, min(b1, b2) - max(a1, a2))
    outer = max(max(b1, b2) - min(a1, a2), smooth)

    return inner / outer

  def __token_similiarity(self, A:Token, B:Token) -> float:
    # ratio = fuzz.ratio(A.text.strip().lower(), B.text.strip().lower())
    ratio = Levenshtein.normalized_similarity(A.tokens, B.tokens)
    iou = self.__token_iou(
      A, B,
      self.__HYPERPARAMETERS["token_iou_padding"]
    )
    return (ratio + iou)/2
    
  def __combine(
    self, 
    A:list[Token], 
    B:list[Token], 
    search_range_time:int = 1.5, 
    threshold:float = 0.5,
    tolerance:float = 0.3 
  ):
    if not A: return B
    
    orphan_tokens = []
    new_token_group = []
    tail = []
    for t in B:
      start = A[-1].start + tolerance
      if not t.is_word and t.end > start: tail.append(t) 
      elif t.is_word and  t.start > start: tail.append(t)
      else: new_token_group.append([t]) 
    
    idx_A , idx_group = -1, 0
    while True:
      idx_A += 1
      if idx_A >= len(A) or idx_group >= len(new_token_group): break

      a = A[idx_A]   
      similarities = []
      for i in range(idx_group, len(new_token_group)):
        token = new_token_group[i][0]
        if token.start < a.start - search_range_time: continue
        elif token.start > a.start + search_range_time: break
        elif not token.is_word: similarities.append((i, 0))
        else: 
          similarity = self.__token_similiarity(a, token)
          similarities.append((i, similarity))

      if not similarities:
        orphan_tokens.append(a)
        continue

      maxarg = max(range(len(similarities)), key=lambda x: similarities[x][1])
      if similarities[maxarg][1] < threshold:
        orphan_tokens.append(a)
        continue
      idx = similarities[maxarg][0]
      new_token_group[idx].append(a)
      idx_group = similarities[0][0] + 1

    for i in range(idx_A, len(A)):
      orphan_tokens.append(A[i])
      
    tokens = [max(tk, key=lambda x: x.probability) for tk in new_token_group] 
    tokens_idx = 0
    orphan_idx = 0
    while True:
      if orphan_idx >= len(orphan_tokens): break
      t = orphan_tokens[orphan_idx]
      if tokens_idx >= len(tokens):
        tokens.append(t)
        orphan_idx += 1
        continue
      token = tokens[tokens_idx]
      if token.is_word and token.start > t.start:
        tokens.insert(tokens_idx, t)
        orphan_idx += 1
      elif not token.is_word and token.end > t.end:
        tokens.insert(tokens_idx, t)
        orphan_idx += 1
      else:
        tokens_idx += 1

    tokens.extend(tail)
    return tokens

  def __refine(
    self,
    audio:np.ndarray,
    time_offset:float,
    token:list[Token],
    tolerance:float = 0.5
  ):
    prev_audio_start = max(0, len(audio) - self.__MAX_PREV_TIME*self._SAMPLE_RATE)
    new_time_offset = time_offset + prev_audio_start/self._SAMPLE_RATE

    completed_words = [t for t in token if t.end < new_time_offset - tolerance]
    prev_recog = [t for t in token if t not in completed_words and t.is_word]
    prev_audio = audio[prev_audio_start:]

    return completed_words, prev_recog, prev_audio, new_time_offset

  @BaseLogger.object
  def __segment_scent(
    self,
    completed_words:list[Token],
    language:str,
    order:int,
    base_logger:Logger
  ):
    tokenizer = Tokenizer.get_tokenizer(language)
    
    completed_dict = {}

    start_idx = 0
    if tokenizer is None:
      # 그냥 온점으로 문장 나누자. 그게 더 좋아 보이네
      base_logger.warning(f"Tokenizer not found for {language}.")
      start_idx = 0
      for idx in range(len(completed_words)):
        word = completed_words[idx]
        if not word.is_word:
          completed_dict[order] = Sentence(
            [language], 
            word.text,
            completed_words[start_idx:idx + 1]
          )
          start_idx = idx + 1
          order += 1
    else:
      completed_words = [w for w in completed_words if w.is_word]
      text = ""
      for idx in range(len(completed_words)):
        word = completed_words[idx]
        text += word.text
        
      start_idx = 0
      scents = tokenizer.segment(text)
      for scent in scents[:-1]:
        len_scent = len(scent)
        for idx in range(start_idx, len(completed_words)):
          len_scent -= len(completed_words[idx].text)
          if len_scent <= 0:
            end_idx = idx + 1
            break
        else:
          end_idx = len(completed_words)

        language = list(set(
          w.lang for w in completed_words[start_idx:end_idx]
        ))

        completed_dict[order] = Sentence(
          language,
          scent,
          completed_words[start_idx:end_idx]
        )
        start_idx = end_idx
        order += 1
    return completed_dict, completed_words[start_idx:], order

