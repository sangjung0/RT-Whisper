import os
import sys

WORKDIR = os.environ["CONTAINER_WORK_DIR"]
os.chdir(WORKDIR)
print(f"Current Python version: {sys.version}")

from pathlib import Path

from sj_ai_utils.datasets.esic_v1 import ESICv1Dataset
from sj_ai_utils.datasets.hugging_face import ZerothKorean
from sj_ai_utils.datasets.hugging_face import KSPonSpeech
from sj_ai_utils.datasets.l_hotse import VoxPopuli
from sj_ai_utils.datasets.l_hotse import Tedlium
from sj_ai_utils.datasets.l_hotse import LibriSpeech
from sj_utils.file.yaml import read_yaml

from rt_whisper_optimizer import Optimizer

VOX_POPULI_PATH = f"{WORKDIR}/.datasets/vox_populi"
TEDLIUM_PATH = f"{WORKDIR}/.datasets/tedlium"
LIBRI_PATH = f"{WORKDIR}/.datasets/libri_speech"
ESIC_PATH = f"{WORKDIR}/test/performance_test/data/esic_train.json"

load_esic = lambda: ESICv1Dataset.load(Path(ESIC_PATH)).sample(-1)
load_zeroth_korean = lambda: ZerothKorean().train().sample(1829)
load_vox_populi = lambda: VoxPopuli(Path(VOX_POPULI_PATH)).load_dev_asr_en().sample(207)
load_tedlium = lambda: Tedlium(Path(TEDLIUM_PATH)).load_dev().sample(-1)
load_libri_clean = lambda: LibriSpeech(Path(LIBRI_PATH)).load_dev_clean().sample(1268)
load_libri_other = lambda: LibriSpeech(Path(LIBRI_PATH)).load_dev_other().sample(1511)
load_ks_pon_speech = lambda: KSPonSpeech().train().sample(2378)

params = [
    # ESIC
    {
        "name": "ESIC",
        "storage": f"{WORKDIR}/.storage/esic/3s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_esic,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/esic/20251003/en/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "ESIC",
        "storage": f"{WORKDIR}/.storage/esic/2s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_esic,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/esic/20251003/en/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "ESIC",
        "storage": f"{WORKDIR}/.storage/esic/1s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_esic,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/esic/20251003/en/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # Zeroth Korean
    {
        "name": "Zeroth Korean",
        "storage": f"{WORKDIR}/.storage/zeroth_korean/3s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_zeroth_korean,
        "language": "ko",
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/zeroth_korean/20251003/ko/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/ko/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "Zeroth Korean",
        "storage": f"{WORKDIR}/.storage/zeroth_korean/2s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_zeroth_korean,
        "language": "ko",
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/zeroth_korean/20251003/ko/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/ko/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "Zeroth Korean",
        "storage": f"{WORKDIR}/.storage/zeroth_korean/1s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_zeroth_korean,
        "language": "ko",
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/zeroth_korean/20251003/ko/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/ko/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # KSPonSpeech
    {
        "name": "KSPonSpeech",
        "storage": f"{WORKDIR}/.storage/kspon_speech/3s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_ks_pon_speech,
        "language": "ko",
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/kspon_speech/20251003/ko/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/ko/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "KSPonSpeech",
        "storage": f"{WORKDIR}/.storage/kspon_speech/2s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_ks_pon_speech,
        "language": "ko",
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/kspon_speech/20251003/ko/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/ko/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "KSPonSpeech",
        "storage": f"{WORKDIR}/.storage/kspon_speech/1s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_ks_pon_speech,
        "language": "ko",
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/kspon_speech/20251003/ko/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/ko/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # VoxPopuli
    {
        "name": "Vox populi",
        "storage": f"{WORKDIR}/.storage/vox_populi/3s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_vox_populi,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/vox_populi/20251003/en/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "vox populi",
        "storage": f"{WORKDIR}/.storage/vox_populi/2s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_vox_populi,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/vox_populi/20251003/en/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "vox populi",
        "storage": f"{WORKDIR}/.storage/vox_populi/1s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_vox_populi,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/vox_populi/20251003/en/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # Tedlium
    {
        "name": "Tedlium",
        "storage": f"{WORKDIR}/.storage/tedlium/3s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_tedlium,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/tedlium/20251003/en/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "Tedlium",
        "storage": f"{WORKDIR}/.storage/tedlium/2s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_tedlium,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/tedlium/20251003/en/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "Tedlium",
        "storage": f"{WORKDIR}/.storage/tedlium/1s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_tedlium,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/tedlium/20251003/en/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # LibriSpeech clean
    {
        "name": "LibriSpeech clean",
        "storage": f"{WORKDIR}/.storage/libri_clean/3s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_libri_clean,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/libri_clean/20251003/en/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "LibriSpeech clean",
        "storage": f"{WORKDIR}/.storage/libri_clean/2s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_libri_clean,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/libri_clean/20251003/en/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "LibriSpeech clean",
        "storage": f"{WORKDIR}/.storage/libri_clean/1s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_libri_clean,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/libri_clean/20251003/en/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # LibriSpeech other
    {
        "name": "LibriSpeech other",
        "storage": f"{WORKDIR}/.storage/libri_other/3s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_libri_other,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/libri_other/20251003/en/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "LibriSpeech other",
        "storage": f"{WORKDIR}/.storage/libri_other/2s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_libri_other,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/libri_other/20251003/en/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "LibriSpeech other",
        "storage": f"{WORKDIR}/.storage/libri_other/1s",
        "study": f"{WORKDIR}/test/optimize/study",
        "dataset": load_libri_other,
        "output": [
            f"{WORKDIR}/test/optimize/hyperparameters/libri_other/20251003/en/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            f"{WORKDIR}/test/optimize/study_param/20251003/en/1s/step2_1s-96k-cpm.yaml",
        ],
    },

]

for param in params:
    name = param["name"]
    storage = Path(param["storage"])
    study = Path(param["study"])
    datasets = param["dataset"]()
    output = param["output"]
    study_instruction = param["study_instruction"]
    print(f"{name}: {len(datasets)}")

    language = param.get("language", None)

    for op, si in zip(output, study_instruction):
        op, si = Path(op), Path(si)
        instruction = read_yaml(si)
        if language is not None:
            instruction["optimizer"]["language"] = language
        optimizer = Optimizer(study, op, instruction, cache_storage=storage)
        optimizer.optimize(datasets, log_step=100, plot_history=False)

    del datasets
