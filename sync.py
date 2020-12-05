import ray

from scheduler import Scheduler
from work import Work


@ray.remote
class Sync(Scheduler):
    def __init__(self, file: str, work: Work):
        Scheduler.__init__(self, file, work)

    def calc(self, work: Work):
        self.sgd(work)
        return
