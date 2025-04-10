from RTWhisper.data import Context
from RTWhisper import Pipeline

class Classifier(Pipeline):
  def __init__(
    self,
    MAX_PREV_SC:int,
  ):
    super().__init__()
    self.__MAX_PREV_SC = MAX_PREV_SC

  def _get_prev_timestamps(self, timestamps:list[dict], anchor:int):
    prev_timestamps = []
    t_index = 0
    while t_index < len(timestamps) and timestamps[t_index]["end"] < anchor:
      t_index += 1
    if t_index < len(timestamps) and timestamps[t_index]["start"] < anchor:
      t = timestamps[t_index]
      prev_timestamps.append({"start": 0, "end": t["end"] - anchor})
      t_index += 1
    while t_index < len(timestamps):
      t = timestamps[t_index]
      prev_timestamps.append({"start": t["start"] - anchor, "end": t["end"] - anchor})
      t_index += 1
    return prev_timestamps

  def process(self, context:Context) -> None:
    audio = context.audio
    processed_audio = context.processed_audio
    sc_offset = context.sc_offset
    tokens = context.tokens
    conditions = context.processed_timestamp_conditions
    prev_audio_sc = context.prev_audio_sc
    timestamps = context.timestamps

    if processed_audio is None:
      context.prev_timestamps = []
      context.prev_audio_sc = 0
      context.sc_offset = sc_offset + len(audio) + prev_audio_sc
      context.completed_words = context.tokens
      context.prev_recog = []
      context.prev_processed_audio = None
      return

    prev_audio_start = max(0, len(processed_audio) - self.__MAX_PREV_SC["default"])
    anchor = 0
    for start, end, offset in conditions:
      if start <= prev_audio_start <= end:
        anchor = offset + prev_audio_start
        break

    prev_audio_sc = prev_audio_sc + len(audio) - anchor 
    sc_offset = sc_offset + anchor
    completed_words = [t for t in tokens if t.end < sc_offset]
    prev_recog = [t for t in tokens if t not in completed_words and t.is_word]
    prev_audio = processed_audio[prev_audio_start:]
    prev_timestamps = self._get_prev_timestamps(timestamps, anchor)

    context.prev_timestamps = prev_timestamps
    context.prev_audio_sc = prev_audio_sc
    context.sc_offset = sc_offset
    context.completed_words = completed_words
    context.prev_recog = prev_recog
    context.prev_processed_audio = prev_audio