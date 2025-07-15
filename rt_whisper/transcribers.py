from pathlib import Path

from sj_utils.collection_utils import SafetyDict

from rt_whisper.core import logger
from rt_whisper.models import SileroVad, Whisper
from rt_whisper.pipeline import Pipeline
from rt_whisper.processors import ASR
from rt_whisper.processors.vad.v2 import VAD
from rt_whisper.composer import SimpleComposer
from rt_whisper.utils import init_hyperparameter, whisper_embed


def get_transcriber(
    hyperparameter: SafetyDict | Path | str | None = None,
):
    hyperparameter = init_hyperparameter(hyperparameter)

    whisper = Whisper(hyperparameter["whisper"]["model_options"])
    transcribe = lambda audio, language, prompt: whisper.transcribe(
        audio, language, prompt, hyperparameter["whisper"]["transcribe_options"]
    )
    silero_vad = SileroVad(
        Whisper.sample_rate, hyperparameter["silero_vad"]["model_options"]
    )
    vad = lambda audio: silero_vad.run(
        audio, hyperparameter["silero_vad"]["run_options"]
    )

    worker_groups = [
        [
            VAD(vad=vad, logger=logger),
            ASR(
                transcriber=transcribe,
                embed=whisper_embed(),
                sample_rate=Whisper.sample_rate,
                within_eos=True,
                max_overlap_duration=hyperparameter["asr"]["max_overlap_duration"],
                logger=logger,
            ),
        ],
        # NOTE SimpleComposer는 로거를 추가하지 않음
        [SimpleComposer()],
    ]

    pipeline = Pipeline()
    pipeline.init(workers=worker_groups)
    return pipeline
