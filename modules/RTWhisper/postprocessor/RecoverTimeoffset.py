from RTWhisper import Pipeline
from RTWhisper.data import Context

class RecoverTimeoffset(Pipeline):
  def __find_condition(
    self, 
    c_index:int,
    conditions:list[dict[str, int]], 
    sample_count:int
  ):
    while c_index < len(conditions) or c_index >= 0:
      c = conditions[c_index]
      start = c["start"]
      end = c["end"]
      offset = c["offset"]
      if start <= sample_count <= end:
        return c_index, offset
      elif sample_count < start:
        c_index -= 1
      else:
        c_index += 1
    
    raise ValueError("Condition not found")

  def can_process(self, context: Context) -> bool:
    if context.merged_processed_audio_sc == 0: return False
    return (
      context.merged_candidate_tokens, 
      context.timestamps, 
      context.prev_timestamps, 
      context.prev_timestamps_mapping, 
      context.prev_audio_sc
    )

  def compute_process(self, param:tuple) -> tuple:
    tokens, timestamps, prev_timestamps, prev_timestamps_mapping, prev_sc = param

    adjusted_timestamps = [{"start": t["start"] + prev_sc, "end": t["end"] + prev_sc} for t in timestamps]

    timestamps_mapping = []
    prev_end = prev_timestamps_mapping[-1]['end'] if prev_timestamps_mapping else 0
    for ts in adjusted_timestamps:
      duration = ts["end"] - ts["start"]
      end = prev_end + duration
      timestamps_mapping.append(
        {"start": prev_end, "end": end, "offset": ts["end"] - end}
      )
      prev_end = end
      
    merged_timestamps_mapping = prev_timestamps_mapping + timestamps_mapping
    merged_timestamps = prev_timestamps + adjusted_timestamps

    c_index = 0
    for token in tokens:
      c_index, offset = self.__find_condition(c_index, merged_timestamps_mapping, token.start)
      token.start = token.start + offset
      c_index, offset = self.__find_condition(c_index, merged_timestamps_mapping, token.end)
      token.end = token.end + offset

    return merged_timestamps, merged_timestamps_mapping #, tokens

  def apply_process(self, context: Context, result:tuple) -> None:
    merged_timestamps, merged_timestamps_mapping = result
    context.merged_timestamps = merged_timestamps
    context.merged_timestamps_mapping = merged_timestamps_mapping
    # context.merged_candidate_tokens = tokens
    
    