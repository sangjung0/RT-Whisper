from logging import Logger
import re
from typing import Generator
import numpy as np

from RTWhisper import BaseLogger
from RTWhisper import Settings

from .Service import Service
from .Sentence import Sentence
from .Params import SentenceParams, SentenceReturn

class SentenceService(Service):
  def __init__(
    self,
    max_prev_sent:int = Settings.MODEL_SENTENCE_MAX_PREV_SENTENCE,
  ):
    super().__init__()

    self.__MAX_PREV_SENT = max_prev_sent
  
  @BaseLogger.object
  def transcribe(
    self,
    sp:SentenceParams,
    base_logger:Logger = None,
  ):

    prompt = sp.prompt if sp.prompt else ""
    prompt = f"{prompt} {sp.prev_sentence.text}" if sp.prev_sentence else prompt

    audio = sp.audio if sp.prev_audio is None else np.concatenate([sp.prev_audio, sp.audio])

    segments, info = self._whisper.transcribe(audio, sp.language, prompt)
    language = info.language
    sentences = self._segment_word(segments, language, sp.time_offset)
    
    (prev_audio, 
     time_offset,
     completed_dict, 
     order, 
     prev_sentence, 
     prev_recog) = self.__refine(audio, 
                               sp.time_offset, 
                               sp.order, 
                               sentences)

    result = SentenceReturn(
      completed_dict=completed_dict,
      order=order,
      time_offset=time_offset,
      prev_audio=prev_audio,
      prev_sentence=prev_sentence,
      prev_recog=prev_recog
    )

    # logger_service.info("-" * 20)
    # logger_service.info(f"Completed sentences: {[(key, value.text) for key, value in completed_dict.items()]}")
    # logger_service.info(f"Pre recog sentences: {[p.text for p in pre_recog]}")
    
    return result, audio

  def __refine(
    self,
    audio:np.ndarray,
    time_offset:float,
    order:int,
    sentences:list[Sentence]
  ):
  
    prev_index = self.__MAX_PREV_SENT + 1

    completed, prev_recog = (
      sentences[:-prev_index], 
      sentences[-prev_index:]
    )

    if completed:
      prev_sentence = completed[-1]
    else:
      prev_sentence = None
    
    completed_dict = {}
    for sentence in completed:
      completed_dict[order] = sentence
      order += 1

    if completed and completed[-1].words:
      word = completed[-1].words[-1]
      start = int((word.end - time_offset) * self._SAMPLE_RATE)
      audio = audio[start:]
      time_offset = time_offset + len(audio) / self._SAMPLE_RATE

    return audio, time_offset, completed_dict, order, prev_sentence, prev_recog

