from rt_whisper.abstracts import Worker
from rt_whisper.data import TokenState, Param, Result


class Pipeline:
    def __init__(self):
        super().__init__()
        self._pipeline = []

    def init(self, workers: list[list[Worker]]):
        if (
            not isinstance(workers, list)
            or not all(isinstance(group, list) for group in workers)
            or not all(
                all(isinstance(worker, Worker) for worker in group) for group in workers
            )
        ):
            raise TypeError("Workers must be a list of lists.")

        new_workers = []
        for worker_group in workers:
            new_workers.append(
                [
                    worker_group,
                    list(reversed(worker_group)),
                    list(reversed(worker_group)),
                ]
            )
        self._workers = workers
        self._pipeline = new_workers

    def process(self, param: Param) -> Result:
        if not self._pipeline:
            raise RuntimeError("Pipeline is not initialized with workers.")

        context = TokenState()
        for worker in self._workers:
            for w in worker:
                w._register_state(context)

        context.bind(param)

        for worker_group in self._pipeline:
            for worker in worker_group[0]:
                worker.process(context)
            for worker in worker_group[1]:
                worker.post_process(context)
            for worker in worker_group[2]:
                worker.context_build(context)

        return context.extract()
