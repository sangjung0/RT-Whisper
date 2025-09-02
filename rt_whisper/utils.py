import torch

from pathlib import Path
from transformers import WhisperTokenizer, WhisperModel
from functools import lru_cache

from sj_utils.collection import SafetyDict
from sj_utils.file.yaml import read_yaml
from sj_utils.decorator import lru_dict_cache

from rt_whisper.core import hyperparameter as default_hyperparameter, config
from rt_whisper.models import Whisper, SileroVad, BoundaryWordFilter


def init_hyperparameter(
    hyperparameter: SafetyDict | Path | str | None = None,
):
    if hyperparameter is None:
        hyperparameter = default_hyperparameter
    elif isinstance(hyperparameter, str) or isinstance(hyperparameter, Path):
        hyperparameter = SafetyDict(read_yaml(Path(hyperparameter)))
    elif not isinstance(hyperparameter, SafetyDict):
        raise TypeError(
            f"Expected hyperparameter to be SafetyDict, Path, or str, got {type(hyperparameter)}"
        )

    return hyperparameter


@lru_cache(maxsize=1)
def whisper_embed(model_size: str = config.rt_whisper.huggingface.path):
    tokenizer = WhisperTokenizer.from_pretrained(model_size)
    embedding_table = _whisper_embedding_weight(model_size)

    @lru_cache(maxsize=4096)
    def embed(text: str):
        tokens = torch.tensor(tokenizer.encode(text))
        if len(tokens) == 0:
            return torch.zeros(embedding_table.shape[1])
        embedding = embedding_table[tokens]

        return embedding.mean(dim=0)

    return embed


def _whisper_embedding_weight(
    model_size: str = config.rt_whisper.huggingface.path,
):
    model = WhisperModel.from_pretrained(model_size)
    embedding = model.decoder.embed_tokens.weight.detach().cpu().clone()

    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    return embedding


@lru_dict_cache()
def get_whisper(options: dict = {}) -> Whisper:
    return Whisper(options)


@lru_dict_cache()
def get_silero_vad(sample_rate: int, options: dict = {}) -> SileroVad:
    return SileroVad(sample_rate, options)


@lru_cache(maxsize=1)
def boundary_word_filter(path: Path) -> BoundaryWordFilter:
    return BoundaryWordFilter.load(path)


__all__ = [
    "init_hyperparameter",
    "whisper_embed",
    "get_whisper",
    "get_silero_vad",
    "boundary_word_filter",
]
