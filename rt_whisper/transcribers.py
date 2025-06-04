from pathlib import Path

from rt_whisper.composer.simple_composer import SimpleComposer
from rt_whisper.core.state import hyperparameter as default_hyperparameter, config
from rt_whisper.util import ReadYaml, SafetyDict
from rt_whisper.models import SileroVad, Whisper
from rt_whisper.pipeline import Pipeline


def get_transcriber(
    model_size: str = config.rt_whisper.model_size,
    model_device: str = config.rt_whisper.model_device,
    model_compute_type: str = config.rt_whisper.model_compute_type,
    model_beam_size: int = config.rt_whisper.model_beam_size,
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
            SileroVad(
                sample_rate=model_sample_rate,
            ),
            Whisper(
                beam_size=model_beam_size,
                model_size=model_size,
                device=model_device,
                compute_type=model_compute_type,
                sample_rate=model_sample_rate,
                within_eos=True,
            ),
        ],
        [
            SimpleComposer(),
        ],
    ]

    pipeline = Pipeline.get_instance()
    pipeline.init(workers=worker_groups)
    return pipeline
