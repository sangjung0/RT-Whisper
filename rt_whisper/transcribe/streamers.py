from pathlib import Path
from whisper.tokenizer import get_tokenizer

from rt_whisper.composer import Composer
from rt_whisper.core.state import config
from rt_whisper.filters import DurationFilter, PositionWeightedFilter, ProbabilityFilter
from rt_whisper.models import BatchedWhisper, SileroVad, Whisper
from rt_whisper.processors.asr import ASR, AsyncASR
from rt_whisper.processors.vad.v1 import VAD as VADv1
from rt_whisper.processors.vad.v2 import VAD as VADv2
from rt_whisper.selector import Selector

from .pipeline import Pipeline
from .service import get_hyperparameter


def get_token_streamer(
    model_sample_rate: int = config.rt_whisper.model_sample_rate,
    hyperparameter_path: Path | str | None = None,
):
    hyperparameter = get_hyperparameter(hyperparameter_path)
    worker_groups = [
        [
            VADv1(vad=SileroVad().run),
            ASR(
                transcriber=Whisper().transcribe,
                tokenizer_encoder=get_tokenizer(multilingual=True).encode,
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
    hyperparameter_path: Path | str | None = None,
):
    hyperparameter = get_hyperparameter(hyperparameter_path)
    worker_groups = [
        [
            VADv2(vad=SileroVad().run),
            ASR(
                transcriber=Whisper().transcribe,
                tokenizer_encoder=get_tokenizer(multilingual=True).encode,
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


def get_batched_token_streamer_with_bad_v2(
    model_sample_rate: int = config.rt_whisper.model_sample_rate,
    hyperparameter_path: Path | str | None = None,
):
    hyperparameter = get_hyperparameter(hyperparameter_path)
    worker_groups = [
        [
            VADv2(vad=SileroVad().run),
            AsyncASR(
                transcriber=BatchedWhisper().transcribe,
                tokenizer_encoder=get_tokenizer(multilingual=True).encode,
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
