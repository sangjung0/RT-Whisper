from RTWhisper.data import Context
from RTWhisper import Pipeline

class SentenceClassifier(Pipeline):

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

  def process(self, context: Context):
    audio = context.audio
    processed_audio = context.processed_audio
    sc_offset = context.sc_offset
    completed = context.completed
    prev_audio_sc = context.prev_audio_sc
    timestamps = context.timestamps
    order = context.order

    if processed_audio is None:
      context.prev_audio_sc = prev_audio_sc + len(audio)
      return

    anchor = (completed[order - 1].tokens[-1].end - sc_offset) if completed else timestamps[0]["start"]

    prev_audio_start = 0
    for ts in timestamps:
      if ts["start"] <= anchor <= ts["end"]:
        prev_audio_start += anchor - ts["start"]
        break
      prev_audio_start += (ts["end"] - ts["start"])
    
    prev_audio_sc = prev_audio_sc + len(audio) - anchor
    sc_offset = sc_offset + anchor
    prev_audio = processed_audio[prev_audio_start:]
    prev_timestamps = self._get_prev_timestamps(timestamps, anchor)

    context.prev_timestamps = prev_timestamps
    context.prev_audio_sc = prev_audio_sc
    context.sc_offset = sc_offset
    context.prev_processed_audio = prev_audio