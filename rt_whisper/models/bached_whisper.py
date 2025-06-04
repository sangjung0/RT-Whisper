from faster_whisper import BatchedInferencePipeline

from data import Context

from .whisper.whisper import Whisper


class BatchedWhisper(Whisper):
    def __init__(
        self,
        beam_size: int = Settings.MODEL_BEAM_SIZE,
        batch_size: int = Settings.MODEL_BATCH_SIZE,
        model_size: str = Settings.MODEL_SIZE,
        device: str = Settings.MODEL_DEVICE,
        compute_type: str = Settings.MODEL_COMPUTE_TYPE,
    ):
        super().__init__(beam_size, model_size, device, compute_type)
        self.__BATCH_SIZE = batch_size

        self.__batched_model = BatchedInferencePipeline(self.__model)

    def process(self, context: Context):
        audio = context.vad_chunk
        language = context.language

        if isinstance(audio, list) and len(audio) > self.__BATCH_SIZE:
            raise ValueError("The maximum number of audio files is 8.")

        segments, info = self.__batched_model.transcribe(
            audio,
            batch_size=self.__BATCH_SIZE,
            beam_size=self.__BEAM_SIZE,
            language=language,
            word_timestamps=True,
            vad_filter=False,
        )

        context.tokens = segments
        context.language = info.language
