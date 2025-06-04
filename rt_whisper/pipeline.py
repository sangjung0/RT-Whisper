from rt_whisper.abstracts import Singleton, Worker
from rt_whisper.data import Context, Param, Result


class Pipeline(Singleton):
    def __init__(self):
        super().__init__()
        self.__workers = []

    def init(self, workers: list[list[Worker]]):
        if (
            not isinstance(workers, list)
            or not all(isinstance(group, list) for group in workers)
            or not all(
                all(isinstance(worker, Worker) for worker in group) for group in workers
            )
        ):
            raise TypeError("Workers must be a list of lists.")
        self.__workers = workers

    def process(self, param: Param) -> Result:
        if not self.__workers:
            raise RuntimeError("Pipeline is not initialized with workers.")

        context = Context()
        context.bind(param)

        for worker_group in self.__workers:
            for worker in worker_group:
                worker.process(context)
            for worker in reversed(worker_group):
                worker.post_process(context)

        return context.extract()
