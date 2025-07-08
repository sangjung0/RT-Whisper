from pathlib import Path

from sj_utils.collection_utils import SafetyDict

from rt_whisper.models import SileroVad, Whisper
from rt_whisper.core.state import config
from rt_whisper.pipeline import Pipeline
from rt_whisper.processors import ASR
from rt_whisper.processors.vad.v2 import VAD
from rt_whisper.composer import SimpleComposer
from rt_whisper.utils import init_hyperparameter, whisper_embed


def get_transcriber(
    model_sample_rate: int = config.rt_whisper.model_sample_rate,
    hyperparameter: SafetyDict | Path | str | None = None,
):

    hyperparameter = init_hyperparameter(hyperparameter)

    worker_groups = [
        [
            VAD(vad=SileroVad().run),
            ASR(
                transcriber=Whisper().transcribe,
                embed=whisper_embed(),
                sample_rate=model_sample_rate,
                within_eos=True,
                max_overlap_duration=hyperparameter["max_overlap_duration"],
            ),
        ],
        [SimpleComposer()],
    ]

    pipeline = Pipeline()
    pipeline.init(workers=worker_groups)
    return pipeline
