from rt_whisper.abstracts import Worker
from rt_whisper.data import TokenState, Param, Result
from sj_utils.decorator_utils import singleton


@singleton
class Pipeline:
    def __init__(self):
        super().__init__()
        self.__pipeline = []

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
        self.__workers = workers
        self.__pipeline = new_workers

    def process(self, param: Param) -> Result:
        if not self.__pipeline:
            raise RuntimeError("Pipeline is not initialized with workers.")

        context = TokenState()
        for worker in self.__workers:
            for w in worker:
                w._register_state(context)

        context.bind(param)

        for worker_group in self.__pipeline:
            for worker in worker_group[0]:
                worker.process(context)
            for worker in worker_group[1]:
                worker.post_process(context)
            for worker in worker_group[2]:
                worker.context_build(context)

        return context.extract()
