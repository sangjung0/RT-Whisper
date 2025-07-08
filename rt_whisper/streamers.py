from pathlib import Path

from sj_utils.collection_utils import SafetyDict

from rt_whisper.composer import Composer
from rt_whisper.core.state import config
from rt_whisper.filters import DurationFilter, PositionWeightedFilter, ProbabilityFilter
from rt_whisper.models import SileroVad, Whisper
from rt_whisper.processors import ASR
from rt_whisper.processors.vad.v1 import VAD as VADv1
from rt_whisper.processors.vad.v2 import VAD as VADv2
from rt_whisper.selector import Selector
from rt_whisper.pipeline import Pipeline
from rt_whisper.utils import init_hyperparameter, whisper_embed


def get_token_streamer(
    model_sample_rate: int = config.rt_whisper.model_sample_rate,
    hyperparameter: SafetyDict | Path | str | None = None,
):
    hyperparameter = init_hyperparameter(hyperparameter)

    worker_groups = [
        [
            VADv1(vad=SileroVad().run),
            ASR(
                transcriber=Whisper().transcribe,
                embed=whisper_embed(),
                sample_rate=model_sample_rate,
                within_eos=True,
                max_overlap_duration=hyperparameter["max_overlap_duration"],
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
            Composer(),
        ],
    ]

    pipeline = Pipeline()
    pipeline.init(workers=worker_groups)
    return pipeline


def get_token_streamer_with_vad_v2(
    model_sample_rate: int = config.rt_whisper.model_sample_rate,
    hyperparameter: SafetyDict | Path | str | None = None,
):
    hyperparameter = init_hyperparameter(hyperparameter)

    worker_groups = [
        [
            VADv2(vad=SileroVad().run),
            ASR(
                transcriber=Whisper().transcribe,
                embed=whisper_embed(),
                sample_rate=model_sample_rate,
                within_eos=True,
                max_overlap_duration=hyperparameter["max_overlap_duration"],
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
            Composer(),
        ],
    ]

    pipeline = Pipeline()
    pipeline.init(workers=worker_groups)
    return pipeline
