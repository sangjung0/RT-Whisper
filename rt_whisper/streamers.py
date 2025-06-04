from pathlib import Path

from rt_whisper.classifier import SentenceClassifier, TokenClassifier
from rt_whisper.composer import Composer, SentenceComposer
from rt_whisper.core.state import hyperparameter as default_hyperparameter, config
from rt_whisper.filters import DurationFilter, PositionWeightedFilter, ProbabilityFilter
from rt_whisper.models import SileroVad, Whisper
from rt_whisper.processors import ASR, VAD
from rt_whisper.selector import Selector
from rt_whisper.util import ReadYaml, SafetyDict
from rt_whisper.pipeline import Pipeline


def get_token_streamer(
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
            VAD(vad=SileroVad.get_instance().run),
            ASR(
                transcribe=Whisper.get_instance().transcribe,
                tokenizer_encode=Whisper.get_instance().get_tokenizer().encode,
                sample_rate=model_sample_rate,
                within_eos=True,
            ),
        ],
        [
            PositionWeightedFilter(
                boundary=hyperparameter["weighted_and_offset_token_boundary"]
            ),
            DurationFilter(z_thresh=hyperparameter["duration_filter_z"]),
            ProbabilityFilter(
                z_thresh=hyperparameter["probability_filter"]["z"],
                min_prob=hyperparameter["probability_filter"]["min_prob"],
            ),
            Selector(
                search_range_sc=hyperparameter["selector"]["search_range_sc"],
                threshold=hyperparameter["selector"]["threshold"],
                padding=hyperparameter["selector"]["padding"],
                tolerance=hyperparameter["selector"]["tolerance"],
            ),
            TokenClassifier(max_overlap_size=hyperparameter["max_overlap_size"]),
            Composer(),
        ],
    ]

    pipeline = Pipeline.get_instance()
    pipeline.init(workers=worker_groups)
    return pipeline


def get_sentence_streamer(
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
            SentenceComposer(
                max_prev_sent=hyperparameter["sentence_max_prev_sentence"]
            ),
            SentenceClassifier(),
        ],
    ]

    pipeline = Pipeline.get_instance()
    pipeline.init(workers=worker_groups)
    return pipeline
