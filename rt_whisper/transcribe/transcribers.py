import asyncio
from pathlib import Path
from whisper.tokenizer import get_tokenizer

from rt_whisper.models import BatchedWhisper, SileroVad, Whisper
from rt_whisper.core.state import config
from rt_whisper.processors.asr import ASR, AsyncASR
from rt_whisper.processors.vad.v2 import VAD
from rt_whisper.composer import SimpleComposer

from .pipeline import Pipeline
from .async_pipeline import AsyncPipeline
from .service import get_hyperparameter


def get_transcriber(
    model_sample_rate: int = config.rt_whisper.model_sample_rate,
    hyperparameter_path: Path | str | None = None,
):
    hyperparameter = get_hyperparameter(hyperparameter_path)
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


async def get_batched_transcriber(
    model_sample_rate: int = config.rt_whisper.model_sample_rate,
    hyperparameter_path: Path | str | None = None,
):
    hyperparameter = get_hyperparameter(hyperparameter_path)
    worker_groups = [
        [
            VAD(vad=SileroVad().run),
            AsyncASR(
                transcriber=BatchedWhisper().transcribe,
                tokenizer_encoder=get_tokenizer(multilingual=True).encode,
                sample_rate=model_sample_rate,
                within_eos=True,
                max_overlap_duration=hyperparameter["max_overlap_duration"],
            ),
        ],
        [SimpleComposer()],
    ]
    await BatchedWhisper().run()

    pipeline = AsyncPipeline()
    pipeline.init(workers=worker_groups)
    return pipeline
