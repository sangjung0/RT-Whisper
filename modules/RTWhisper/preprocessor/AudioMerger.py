import numpy as np

from RTWhisper import Pipeline
from RTWhisper.data import Context

class AudioMerger(Pipeline):
  
  def process(self, context:Context):
    audio = context.processed_audio
    prev_audio = context.prev_processed_audio

    if audio is None: return
    elif prev_audio is None: return
    
    context.processed_audio = np.concatenate((prev_audio, audio))