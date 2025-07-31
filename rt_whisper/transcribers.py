from pathlib import Path

from sj_utils.collection import SafetyDict

from rt_whisper.core import logger
from rt_whisper.models import Whisper
from rt_whisper.pipeline import Pipeline
from rt_whisper.processors import ASR
from rt_whisper.processors.vad.v2 import VAD
from rt_whisper.composer import SimpleComposer
from rt_whisper.utils import (
    init_hyperparameter,
    whisper_embed,
    get_silero_vad,
    get_whisper,
)


def get_transcriber(
    hyperparameter: SafetyDict | Path | str | None = None,
):
    hyperparameter = init_hyperparameter(hyperparameter)

    whisper = get_whisper(hyperparameter["whisper"]["model_options"])
    transcribe = lambda audio, language, prompt: whisper.transcribe(
        audio, language, prompt, hyperparameter["whisper"]["transcribe_options"]
    )
    silero_vad = get_silero_vad(
        Whisper.sample_rate, hyperparameter["silero_vad"]["model_options"]
    )
    vad = lambda audio: silero_vad.run(
        audio, hyperparameter["silero_vad"]["run_options"]
    )

    logger.debug("\n✅Creating transcriber with VADv2")
    c_logger = logger.get_child()

    worker_groups = [
        [
            VAD(vad=vad, logger=c_logger),
            ASR(
                transcriber=transcribe,
                embed=whisper_embed(),
                sample_rate=Whisper.sample_rate,
                within_eos=True,
                max_prompt_words=hyperparameter["asr"]["max_prompt_words"],
                max_overlap_duration=hyperparameter["asr"]["max_overlap_duration"],
                logger=c_logger,
            ),
        ],
        # NOTE SimpleComposer는 로거를 추가하지 않음
        [SimpleComposer()],
    ]

    pipeline = Pipeline(c_logger)
    pipeline.init(workers=worker_groups)
    return pipeline

__all__ = [
    "get_transcriber"
]
