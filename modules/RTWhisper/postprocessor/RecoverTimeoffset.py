from RTWhisper import Pipeline
from RTWhisper.data import Context

class RecoverTimeoffset(Pipeline):
  def find_condition(
    self, 
    c_index:int,
    conditions:list[tuple[int, int, int]], 
    sample_count:int
  ):
    while c_index < len(conditions) or c_index >= 0:
      start, end, offset= conditions[c_index]
      if start <= sample_count <= end:
        return c_index, offset
      elif sample_count < start:
        c_index -= 1
      else:
        c_index += 1
    
    raise ValueError("Condition not found")

  def process(self, context: Context):
    if context.processed_audio is None: return
    
    tokens = context.tokens
    timestamps = context.timestamps
    prev_sc = context.prev_audio_sc
    prev_timestamps = context.prev_timestamps

    timestamps = [{"start": t["start"] + prev_sc, "end": t["end"] + prev_sc} for t in timestamps]
    timestamps = prev_timestamps + timestamps
    
    conditions = []
    prev_end = 0
    for t in timestamps:
      duration = t["end"] -t["start"] 
      end = prev_end + duration
      conditions.append(
        (prev_end, end, t["end"] - end )
      )
      prev_end = end

    c_index = 0
    for token in tokens:
      c_index, offset = self.find_condition(c_index, conditions, token.start)
      token.start = token.start + offset
      c_index, offset = self.find_condition(c_index, conditions, token.end)
      token.end = token.end + offset

    # context.tokens = tokens
    context.timestamps = timestamps
    context.processed_timestamp_conditions = conditions


      
      
    

    




    