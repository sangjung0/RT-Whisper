from pathlib import Path

from sj_ai_utils.datasets.esic_v1 import ESICv1Dataset
from sj_ai_utils.datasets.hugging_face import ZerothKorean
from sj_ai_utils.datasets.hugging_face import KSPonSpeech
from sj_ai_utils.datasets.l_hotse import VoxPopuli
from sj_ai_utils.datasets.l_hotse import Tedlium
from sj_ai_utils.datasets.l_hotse import LibriSpeech

from rt_whisper_optimizer import Optimizer

VOX_POPULI_PATH = "/workspaces/dev/.datasets/vox_populi"
TEDLIUM_PATH = "/workspaces/dev/.datasets/tedlium"
LIBRI_PATH = "/workspaces/dev/.datasets/libri_speech"
ESIC_PATH = "/workspaces/dev/test/performance_test/data/esic_train.json"

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
        "storage": "/workspaces/dev/.storage/esic/3s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_esic,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/esic/20250928/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "ESIC",
        "storage": "/workspaces/dev/.storage/esic/2s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_esic,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/esic/20250928/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "ESIC",
        "storage": "/workspaces/dev/.storage/esic/1s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_esic,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/esic/20250928/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # Zeroth Korean
    {
        "name": "Zeroth Korean",
        "storage": "/workspaces/dev/.storage/zeroth_korean/3s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_zeroth_korean,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/zeroth_korean/20250928/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "Zeroth Korean",
        "storage": "/workspaces/dev/.storage/zeroth_korean/2s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_zeroth_korean,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/zeroth_korean/20250928/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "Zeroth Korean",
        "storage": "/workspaces/dev/.storage/zeroth_korean/1s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_zeroth_korean,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/zeroth_korean/20250928/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # KSPonSpeech
    {
        "name": "KSPonSpeech",
        "storage": "/workspaces/dev/.storage/kspon_speech/3s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_ks_pon_speech,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/kspon_speech/20250928/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "KSPonSpeech",
        "storage": "/workspaces/dev/.storage/kspon_speech/2s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_ks_pon_speech,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/kspon_speech/20250928/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "KSPonSpeech",
        "storage": "/workspaces/dev/.storage/kspon_speech/1s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_ks_pon_speech,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/kspon_speech/20250928/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # VoxPopuli
    {
        "name": "Vox populi",
        "storage": "/workspaces/dev/.storage/vox_populi/3s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_vox_populi,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/vox_populi/20250928/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "vox populi",
        "storage": "/workspaces/dev/.storage/vox_populi/2s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_vox_populi,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/vox_populi/20250928/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "vox populi",
        "storage": "/workspaces/dev/.storage/vox_populi/1s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_vox_populi,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/vox_populi/20250928/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # Tedlium
    {
        "name": "Tedlium",
        "storage": "/workspaces/dev/.storage/tedlium/3s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_tedlium,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/tedlium/20250928/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "Tedlium",
        "storage": "/workspaces/dev/.storage/tedlium/2s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_tedlium,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/tedlium/20250928/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "Tedlium",
        "storage": "/workspaces/dev/.storage/tedlium/1s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_tedlium,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/tedlium/20250928/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # LibriSpeech clean
    {
        "name": "LibriSpeech clean",
        "storage": "/workspaces/dev/.storage/libri_clean/3s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_libri_clean,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/libri_clean/20250928/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "LibriSpeech clean",
        "storage": "/workspaces/dev/.storage/libri_clean/2s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_libri_clean,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/libri_clean/20250928/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "LibriSpeech clean",
        "storage": "/workspaces/dev/.storage/libri_clean/1s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_libri_clean,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/libri_clean/20250928/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/1s/step2_1s-96k-cpm.yaml",
        ],
    },
    # LibriSpeech other
    {
        "name": "LibriSpeech other",
        "storage": "/workspaces/dev/.storage/libri_other/3s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_libri_other,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/libri_other/20250928/3s/step2_3s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/3s/step2_3s-96k-cpm.yaml",
        ],
    },
    {
        "name": "LibriSpeech other",
        "storage": "/workspaces/dev/.storage/libri_other/2s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_libri_other,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/libri_other/20250928/2s/step2_2s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/2s/step2_2s-96k-cpm.yaml",
        ],
    },
    {
        "name": "LibriSpeech other",
        "storage": "/workspaces/dev/.storage/libri_other/1s",
        "study": "/workspaces/dev/test/optimize/study",
        "dataset": load_libri_other,
        "output": [
            "/workspaces/dev/test/optimize/hyperparameters/libri_other/20250928/1s/step2_1s-96k-cpm",
        ],
        "study_instruction": [
            "/workspaces/dev/test/optimize/study_param/20250928/1s/step2_1s-96k-cpm.yaml",
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

    for op, si in zip(output, study_instruction):
        op, si = Path(op), Path(si)
        optimizer = Optimizer(study, op, si, cache_storage=storage)
        optimizer.optimize(datasets, log_step=100, plot_history=False)

    del datasets

