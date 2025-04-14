import numpy as np

from RTWhisper import Pipeline
from RTWhisper.data import Context

class AudioMerger(Pipeline):
  
  def process(self, context:Context):
    audio = context.processed_audio
    prev_audio = context.prev_processed_audio

    if len(audio) == 0: return
    elif len(prev_audio) == 0 : return
    
    context.processed_audio = np.concatenate((prev_audio, audio))