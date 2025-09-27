from pathlib import Path

from sj_ai_utils.datasets.esic_v1 import ESICv1Dataset

from rt_whisper_optimizer import Optimizer

SAMPLE_SIZE = -1
ESIC_TRAIN = "/workspaces/dev/test/performance_test/data/esic_train.json"

load_esic = lambda: ESICv1Dataset.load(Path(ESIC_TRAIN)).sample(SAMPLE_SIZE)

params = [
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
    }
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
        optimizer.optimize(datasets, log_step=100)
