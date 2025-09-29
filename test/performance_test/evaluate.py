import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/test/performance_test",
]
for path in paths:
    sys.path.append(os.path.abspath(path))
print(f"Current Python version: {sys.version}")

import gc
import torch
from pathlib import Path

from sj_ai_utils.datasets.hugging_face import ZerothKorean
from sj_ai_utils.datasets.l_hotse import VoxPopuli
from sj_ai_utils.datasets.l_hotse import Tedlium
from sj_ai_utils.datasets.l_hotse import LibriSpeech
from sj_ai_utils.datasets.hugging_face import KSPonSpeech
from sj_ai_utils.datasets.esic_v1 import ESICv1Dataset

from common_util import evaluate

MODEL_SIZE = "large-v2"
SAMPLE_SIZE = 1
VOX_POPULI_PATH = "/workspaces/dev/.datasets/vox_populi"
TEDLIUM_PATH = "/workspaces/dev/.datasets/tedlium"
LIBRI_PATH = "/workspaces/dev/.datasets/libri_speech"
ESIC_PATH = "/workspaces/dev/test/performance_test/data/esic_val.json"

load_zeroth_korean = lambda: ZerothKorean().test().sample(SAMPLE_SIZE)
load_vox_populi = (
    lambda: VoxPopuli(Path(VOX_POPULI_PATH)).load_test_asr_en().sample(SAMPLE_SIZE)
)
load_tedlium = lambda: Tedlium(Path(TEDLIUM_PATH)).load_test().sample(SAMPLE_SIZE)
load_libri = lambda: LibriSpeech(Path(LIBRI_PATH)).load_test_clean().sample(SAMPLE_SIZE)
load_ks_pon_speech = lambda: KSPonSpeech().test().sample(SAMPLE_SIZE)
load_esic = lambda: ESICv1Dataset.load(Path(ESIC_PATH)).sample(SAMPLE_SIZE)

PARAMETER = [
    #
    # Baseline and Streaming Whisper
    {
        "name": "Baseline Whisper (zeroth_korean)",
        "test_all": True,
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_baseline-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["whisper"],
    },
    {
        "name": "Baseline Whisper (vox_populi)",
        "test_all": True,
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_baseline-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["whisper"],
    },
    {
        "name": "Baseline Whisper (tedlium)",
        "test_all": True,
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_baseline-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["whisper"],
    },
    {
        "name": "Baseline Whisper (libri)",
        "test_all": True,
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_baseline-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["whisper"],
    },
    {
        "name": "Baseline Whisper (ks_pon_speech)",
        "test_all": True,
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_baseline-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["whisper"],
    },
    {
        "name": "Baseline Whisper (esic)",
        "test_all": True,
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_baseline-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["whisper"],
    },
    #
    # simul whisper
    ## Zeroth Korean
    {
        "name": "Baseline 3s simul whisper (zeroth_korean)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_baseline-3s-simul-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 2s simul whisper (zeroth_korean)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_baseline-2s-simul-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 1s simul whisper (zeroth_korean)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_baseline-1s-simul-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["simul_whisper"],
    },
    #
    ## Vox Populi
    {
        "name": "Baseline 3s simul whisper (vox_populi)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_baseline-3s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 2s simul whisper (vox_populi)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_baseline-2s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 1s simul whisper (vox_populi)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_baseline-1s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["simul_whisper"],
    },
    #
    ## tedlium
    {
        "name": "Baseline 3s simul whisper (tedlium)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_baseline-3s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 2s simul whisper (tedlium)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_baseline-2s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 1s simul whisper (tedlium)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_baseline-1s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["simul_whisper"],
    },
    #
    ## libri
    {
        "name": "Baseline 3s simul whisper (libri)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_baseline-3s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 2s simul whisper (libri)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_baseline-2s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 1s simul whisper (libri)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_baseline-1s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["simul_whisper"],
    },
    #
    ## ks_pon_speech
    {
        "name": "Baseline 3s simul whisper (ks_pon_speech)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_baseline-3s-simul-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 2s simul whisper (ks_pon_speech)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_baseline-2s-simul-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 1s simul whisper (ks_pon_speech)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_baseline-1s-simul-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["simul_whisper"],
    },
    #
    ## esic
    {
        "name": "Baseline 3s simul whisper (esic)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_baseline-3s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 2s simul whisper (esic)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_baseline-2s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["simul_whisper"],
    },
    {
        "name": "Baseline 1s simul whisper (esic)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_baseline-1s-simul-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["simul_whisper"],
    },
    #
    # Streaming Whisper
    ## Zeroth Korean
    {
        "name": "Baseline 3s Streaming Whisper (zeroth_korean)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_baseline-3s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 2s Streaming Whisper (zeroth_korean)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_baseline-2s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 1s Streaming Whisper (zeroth_korean)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_baseline-1s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["whisper_streaming"],
    },
    #
    ## Vox Populi
    {
        "name": "Baseline 3s Streaming Whisper (vox_populi)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_baseline-3s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 2s Streaming Whisper (vox_populi)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_baseline-2s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 1s Streaming Whisper (vox_populi)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_baseline-1s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["whisper_streaming"],
    },
    #
    ## tedlium
    {
        "name": "Baseline 3s Streaming Whisper (tedlium)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_baseline-3s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 2s Streaming Whisper (tedlium)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_baseline-2s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 1s Streaming Whisper (tedlium)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_baseline-1s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["whisper_streaming"],
    },
    #
    ## libri
    {
        "name": "Baseline 3s Streaming Whisper (libri)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_baseline-3s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 2s Streaming Whisper (libri)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_baseline-2s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 1s Streaming Whisper (libri)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_baseline-1s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["whisper_streaming"],
    },
    #
    ## ks_pon_speech
    {
        "name": "Baseline 3s Streaming Whisper (ks_pon_speech)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_baseline-3s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 2s Streaming Whisper (ks_pon_speech)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_baseline-2s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 1s Streaming Whisper (ks_pon_speech)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_baseline-1s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["whisper_streaming"],
    },
    #
    ## esic
    {
        "name": "Baseline 3s Streaming Whisper (esic)",
        "test_all": True,
        "chunk_size": 48000,
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_baseline-3s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 2s Streaming Whisper (esic)",
        "test_all": True,
        "chunk_size": 32000,
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_baseline-2s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["whisper_streaming"],
    },
    {
        "name": "Baseline 1s Streaming Whisper (esic)",
        "test_all": True,
        "chunk_size": 16000,
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_baseline-1s-streaming-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["whisper_streaming"],
    },
    #
    # RT Whisper
    ## Zeroth Korean
    {
        "name": "Evaluate 3s RT Whisper (zeroth_korean)",
        "test_all": True,
        "chunk_size": 48000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/zeroth_korean/3s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/3s/step2_3s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_evaluate-3s-rt-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 2s RT Whisper (zeroth_korean)",
        "test_all": True,
        "chunk_size": 32000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/zeroth_korean/2s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/2s/step2_2s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_evaluate-2s-rt-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 1s RT Whisper (zeroth_korean)",
        "test_all": True,
        "chunk_size": 16000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/zeroth_korean/1s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/1s/step2_1s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/zeroth_korean/test_evaluate-1s-rt-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_zeroth_korean,
        "test_models": ["rt_whisper"],
    },
    #
    ## Vox Populi
    {
        "name": "Evaluate 3s RT Whisper (vox_populi)",
        "test_all": True,
        "chunk_size": 48000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/vox_populi/3s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/3s/step2_3s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_evaluate-3s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 2s RT Whisper (vox_populi)",
        "test_all": True,
        "chunk_size": 32000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/vox_populi/2s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/2s/step2_2s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_evaluate-2s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 1s RT Whisper (vox_populi)",
        "test_all": True,
        "chunk_size": 16000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/vox_populi/1s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/1s/step2_1s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/vox_populi/test_evaluate-1s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_vox_populi,
        "test_models": ["rt_whisper"],
    },
    #
    ## tedlium
    {
        "name": "Evaluate 3s RT Whisper (tedlium)",
        "test_all": True,
        "chunk_size": 48000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/tedlium/3s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/3s/step2_3s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_evaluate-3s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 2s RT Whisper (tedlium)",
        "test_all": True,
        "chunk_size": 32000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/tedlium/2s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/2s/step2_2s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_evaluate-2s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 1s RT Whisper (tedlium)",
        "test_all": True,
        "chunk_size": 16000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/tedlium/1s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/1s/step2_1s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/tedlium/test_evaluate-1s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_tedlium,
        "test_models": ["rt_whisper"],
    },
    #
    ## libri
    {
        "name": "Evaluate 3s RT Whisper (libri)",
        "test_all": True,
        "chunk_size": 48000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/libri/3s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/3s/step2_3s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_evaluate-3s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 2s RT Whisper (libri)",
        "test_all": True,
        "chunk_size": 32000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/libri/2s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/2s/step2_2s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_evaluate-2s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 1s RT Whisper (libri)",
        "test_all": True,
        "chunk_size": 16000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/libri/1s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/1s/step2_1s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/libri/test_evaluate-1s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_libri,
        "test_models": ["rt_whisper"],
    },
    #
    ## ks_pon_speech
    {
        "name": "Evaluate 3s RT Whisper (ks_pon_speech)",
        "test_all": True,
        "chunk_size": 48000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/ks_pon_speech/3s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/3s/step2_3s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_evaluate-3s-rt-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 2s RT Whisper (ks_pon_speech)",
        "test_all": True,
        "chunk_size": 32000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/ks_pon_speech/2s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/2s/step2_2s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_evaluate-2s-rt-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 1s RT Whisper (ks_pon_speech)",
        "test_all": True,
        "chunk_size": 16000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/ks_pon_speech/1s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/1s/step2_1s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/ks_pon_speech/test_evaluate-1s-rt-whisper-{MODEL_SIZE}.json",
        "language": "ko",
        "dataset": load_ks_pon_speech,
        "test_models": ["rt_whisper"],
    },
    #
    ## esic
    {
        "name": "Evaluate 3s RT Whisper (esic)",
        "test_all": True,
        "chunk_size": 48000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/esic/3s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/3s/step2_3s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_evaluate-3s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 2s RT Whisper (esic)",
        "test_all": True,
        "chunk_size": 32000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/esic/2s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/2s/step2_2s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_evaluate-2s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["rt_whisper"],
    },
    {
        "name": "Evaluate 1s RT Whisper (esic)",
        "test_all": True,
        "chunk_size": 16000,
        "use_token_saver_loader": True,
        "use_prompt": False,
        "storage": "/workspaces/dev/.storage/esic/1s",
        "hyperparameter": "/workspaces/dev/test/optimize/esic/hyperparameters/20250917/1s/step2_1s-96k-cpm/001_0_046.yaml",
        "output_path": f"/workspaces/dev/test/performance_test/output/esic/test_evaluate-1s-rt-whisper-{MODEL_SIZE}.json",
        "language": "en",
        "dataset": load_esic,
        "test_models": ["rt_whisper"],
    },
    #
]

for param in PARAMETER:
    name = param["name"]

    output_path = Path(param["output_path"])
    dataset = param["dataset"]()
    test_models = param["test_models"]
    language = param["language"]
    test_all = param["test_all"]

    chunk_size = None
    storage = None
    hyperparameter_path = None
    use_token_saver_loader = None
    use_prompt = None
    if not "whisper" in test_models:
        chunk_size = param["chunk_size"]
    if "rt_whisper" in test_models:
        storage = Path(param["storage"])
        hyperparameter_path = Path(param["hyperparameter"])
        use_token_saver_loader = param["use_token_saver_loader"]
        use_prompt = param["use_prompt"]
        if not hyperparameter_path.exists():
            raise FileNotFoundError(
                f"Hyperparameter path does not exist: {hyperparameter_path}"
            )
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")


    print(f"=== {name} ===")
    print(f"Dataset length: {len(dataset)}")

    description = f"""
    USE_TOKEN_SAVER_LOADER: {use_token_saver_loader}
    USE_PROMPT: {use_prompt}
    CHUNK_SIZE: {chunk_size}

    HYPERPARAMETER: {hyperparameter_path}
    OUTPUT_PATH: {output_path}
    """

    evaluate(
        output_path,
        description,
        dataset,
        test_models,
        MODEL_SIZE,
        chunk_size,
        language=language,
        test_all=test_all,
        storage=storage,
        use_save_loader=use_token_saver_loader,
        use_prompt=use_prompt,
        hyperparameter=hyperparameter_path,
    )

    del dataset

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
