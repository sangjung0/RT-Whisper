from pathlib import Path
from whisper.tokenizer import get_tokenizer

from sj_utils.collection_utils import SafetyDict
from sj_utils.file import ReadYaml

from rt_whisper.models import SileroVad, Whisper
from rt_whisper.core.state import hyperparameter as default_hyperparameter, config
from rt_whisper.pipeline import Pipeline
from rt_whisper.processors import ASR, VAD
from rt_whisper.composer import SimpleComposer


def get_transcriber(
    model_sample_rate: int = config.rt_whisper.model_sample_rate,
    hyperparameter_path: Path | str | None = None,
):
    if hyperparameter_path is None:
        hyperparameter = default_hyperparameter
    else:
        if isinstance(hyperparameter_path, str):
            hyperparameter_path = Path(hyperparameter_path)
        hyperparameter = SafetyDict(ReadYaml(hyperparameter_path).dict)

    worker_groups = [
        [
            VAD(vad=SileroVad().run),
            ASR(
                transcriber=Whisper().transcribe,
                tokenizer_encoder=get_tokenizer(multilingual=True).encode,
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
