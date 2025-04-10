from RTWhisper.data import Context

class Pipeline:
  def process(self, context:Context) -> None:
    raise NotImplementedError("Pipeline process method not implemented")
  