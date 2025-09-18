from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path

from sj_utils.collection import SafetyDict

from rt_whisper.core import logger
from rt_whisper.composer import Composer
from rt_whisper.filters import (
    DurationFilter,
    PositionWeightedFilter,
    ProbabilityFilter,
    DurationMinFilter,
    ProbabilityMinFilter,
)
from rt_whisper.models import Whisper
from rt_whisper.processors.asr import ASR
from rt_whisper.processors.vad.v2 import VAD
from rt_whisper.selector import Selector
from rt_whisper.pipeline import Pipeline
from rt_whisper.archiver import DataSaver, DataLoader
from rt_whisper.utils import (
    init_hyperparameter,
    whisper_embed,
    fasttext_embed,
    get_silero_vad,
    get_whisper,
    boundary_word_filter,
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
        Whisper.SAMPLE_RATE, dict(hyperparameter["silero_vad"]["model_options"])
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
                embed=fasttext_embed(),
                sample_rate=Whisper.SAMPLE_RATE,
                within_eos=True,
                max_prompt_words=int(hyperparameter["asr"]["max_prompt_words"]),
                max_overlap_duration=int(hyperparameter["asr"]["max_overlap_duration"]),
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
                head_model=boundary_word_filter(
                    Path(
                        str(
                            hyperparameter["position_weighted_filter"][
                                "head_model_path"
                            ]
                        )
                    )
                ),
                tail_model=boundary_word_filter(
                    Path(
                        str(
                            hyperparameter["position_weighted_filter"][
                                "tail_model_path"
                            ]
                        )
                    )
                ),
                head_boundary=float(
                    hyperparameter["position_weighted_filter"]["head_boundary"]
                ),
                tail_boundary=float(
                    hyperparameter["position_weighted_filter"]["tail_boundary"]
                ),
                logger=c_logger,
            ),
            DurationMinFilter(
                min_dur=hyperparameter["duration_filter"]["min_dur"],
                logger=c_logger,
            ),
            ProbabilityMinFilter(
                min_prob=hyperparameter["probability_filter"]["min_prob"],
                logger=c_logger,
            ),
            Selector(
                logger=c_logger,
                iou_threshold=hyperparameter["selector"]["iou_threshold"],
                cos_threshold=hyperparameter["selector"]["cos_threshold"],
                padding=hyperparameter["selector"]["padding"],
                m=float(hyperparameter["selector"]["m"]),
                p=float(hyperparameter["selector"]["p"]),
                s=float(hyperparameter["selector"]["s"]),
                window=int(hyperparameter["selector"]["window"]),
                gs_alpha=float(hyperparameter["selector"]["gs_alpha"]),
                st_alpha=float(hyperparameter["selector"]["st_alpha"]),
                token_group_size=int(hyperparameter["selector"]["token_group_size"]),
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
                head_model=boundary_word_filter(
                    Path(
                        str(
                            hyperparameter["position_weighted_filter"][
                                "head_model_path"
                            ]
                        )
                    )
                ),
                tail_model=boundary_word_filter(
                    Path(
                        str(
                            hyperparameter["position_weighted_filter"][
                                "tail_model_path"
                            ]
                        )
                    )
                ),
                head_boundary=float(
                    hyperparameter["position_weighted_filter"]["head_boundary"]
                ),
                tail_boundary=float(
                    hyperparameter["position_weighted_filter"]["tail_boundary"]
                ),
                logger=c_logger,
            ),
            DurationMinFilter(
                min_dur=hyperparameter["duration_filter"]["min_dur"],
                logger=c_logger,
            ),
            ProbabilityMinFilter(
                min_prob=hyperparameter["probability_filter"]["min_prob"],
                logger=c_logger,
            ),
            Selector(
                logger=c_logger,
                iou_threshold=hyperparameter["selector"]["iou_threshold"],
                cos_threshold=hyperparameter["selector"]["cos_threshold"],
                padding=hyperparameter["selector"]["padding"],
                m=float(hyperparameter["selector"]["m"]),
                p=float(hyperparameter["selector"]["p"]),
                s=float(hyperparameter["selector"]["s"]),
                window=int(hyperparameter["selector"]["window"]),
                gs_alpha=float(hyperparameter["selector"]["gs_alpha"]),
                st_alpha=float(hyperparameter["selector"]["st_alpha"]),
                token_group_size=int(hyperparameter["selector"]["token_group_size"]),
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
