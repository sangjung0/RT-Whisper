from transformers import WhisperTokenizer, WhisperModel
import torch

from pathlib import Path

from sj_utils.collection_utils import SafetyDict
from sj_utils.file import ReadYaml

from rt_whisper.core.state import hyperparameter as default_hyperparameter, config


def init_hyperparameter(
    hyperparameter: SafetyDict | Path | str | None = None,
):
    if hyperparameter is None:
        hyperparameter = default_hyperparameter
    elif isinstance(hyperparameter, str) or isinstance(hyperparameter, Path):
        hyperparameter = SafetyDict(ReadYaml(Path(hyperparameter)).dict)
    elif not isinstance(hyperparameter, SafetyDict):
        raise TypeError(
            f"Expected hyperparameter to be SafetyDict, Path, or str, got {type(hyperparameter)}"
        )

    return hyperparameter


def whisper_embed(model_size: str = config.rt_whisper.model_huggingface_path):
    tokenizer = WhisperTokenizer.from_pretrained(model_size)
    embedding_table = _whisper_embedding_weight(model_size)

    def embed(text: str):
        tokens = torch.tensor(tokenizer.encode(text))
        if len(tokens) == 0:
            return torch.zeros(embedding_table.shape[1])
        embedding = embedding_table[tokens]

        return embedding.mean(dim=0)

    return embed


def _whisper_embedding_weight(
    model_size: str = config.rt_whisper.model_huggingface_path,
):
    model = WhisperModel.from_pretrained(model_size)
    embedding = model.decoder.embed_tokens.weight.detach().cpu().clone()

    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    return embedding


__all__ = [
    "init_hyperparameter",
    "whisper_embed",
]
