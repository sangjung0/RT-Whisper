from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path

from sj_utils.collection_utils import SafetyDict

from rt_whisper.core import logger
from rt_whisper.composer import Composer
from rt_whisper.filters import (
    DurationFilter,
    PositionWeightedFilter,
    ProbabilityFilter,
    DurationMinFilter,
)
from rt_whisper.models import Whisper
from rt_whisper.processors import ASR
from rt_whisper.processors.vad.v2 import VAD
from rt_whisper.selector import Selector
from rt_whisper.pipeline import Pipeline
from rt_whisper.optimizer import DataSaver, DataLoader
from rt_whisper.utils import (
    init_hyperparameter,
    whisper_embed,
    get_silero_vad,
    get_whisper,
)

if TYPE_CHECKING:
    pass


def get_token_streamer_saver(
    save_path: Path,
    hyperparameter: SafetyDict | Path | str | None = None,
):
    hyperparameter = init_hyperparameter(hyperparameter)

    whisper = get_whisper(hyperparameter["whisper"]["model_options"])
    transcribe = lambda audio, language, prompt: whisper.transcribe(
        audio, language, prompt, hyperparameter["whisper"]["transcribe_options"]
    )
    silero_vad = get_silero_vad(
        Whisper.SAMPLE_RATE, hyperparameter["silero_vad"]["model_options"]
    )
    vad = lambda audio: silero_vad.run(
        audio, hyperparameter["silero_vad"]["run_options"]
    )

    logger.debug("\n✅Creating token streamer saver")
    c_logger = logger.get_child()

    worker_groups = [
        [
            # NOTE v1에는 로거 추가하지 않음
            VAD(vad=vad, logger=c_logger),
            ASR(
                transcriber=transcribe,
                embed=whisper_embed(),
                sample_rate=Whisper.SAMPLE_RATE,
                within_eos=True,
                max_overlap_duration=hyperparameter["asr"]["max_overlap_duration"],
                logger=c_logger,
            ),
        ],
        [
            DataSaver(
                save_path=save_path,
                logger=c_logger,
            )
        ],
        [
            PositionWeightedFilter(
                boundary=hyperparameter["position_weighted_filter"]["boundary"],
                logger=c_logger,
            ),
            DurationFilter(
                z_thresh=hyperparameter["duration_filter"]["z_thresh"],
                min_dur=hyperparameter["duration_filter"]["min_dur"],
                logger=c_logger,
            ),
            ProbabilityFilter(
                z_thresh=hyperparameter["probability_filter"]["z_thresh"],
                min_prob=hyperparameter["probability_filter"]["min_prob"],
                logger=c_logger,
            ),
            Selector(
                iou_threshold=hyperparameter["selector"]["iou_threshold"],
                cos_threshold=hyperparameter["selector"]["cos_threshold"],
                padding=hyperparameter["selector"]["padding"],
                logger=c_logger,
            ),
            Composer(c_logger),
        ],
    ]

    pipeline = Pipeline(c_logger)
    pipeline.init(workers=worker_groups)
    return pipeline


def get_token_streamer_loader(
    saved_path: Path,
    hyperparameter: SafetyDict | Path | str | None = None,
):
    hyperparameter = init_hyperparameter(hyperparameter)

    logger.debug("\n✅Creating token streamer loader")
    c_logger = logger.get_child()

    worker_groups = [
        [DataLoader(saved_path=saved_path, logger=c_logger)],
        [
            PositionWeightedFilter(
                boundary=hyperparameter["position_weighted_filter"]["boundary"],
                logger=c_logger,
            ),
            DurationFilter(
                z_thresh=hyperparameter["duration_filter"]["z_thresh"],
                min_dur=hyperparameter["duration_filter"]["min_dur"],
                logger=c_logger,
            ),
            ProbabilityFilter(
                z_thresh=hyperparameter["probability_filter"]["z_thresh"],
                min_prob=hyperparameter["probability_filter"]["min_prob"],
                logger=c_logger,
            ),
            Selector(
                iou_threshold=hyperparameter["selector"]["iou_threshold"],
                cos_threshold=hyperparameter["selector"]["cos_threshold"],
                padding=hyperparameter["selector"]["padding"],
                logger=c_logger,
            ),
            Composer(c_logger),
        ],
    ]

    pipeline = Pipeline(c_logger)
    pipeline.init(workers=worker_groups)
    return pipeline


__all__ = [
    "get_token_streamer_saver",
    "get_token_streamer_loader",
]
