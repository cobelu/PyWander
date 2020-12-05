import ray

from scheduler import Scheduler
from work import Work


@ray.remote
class Sync(Scheduler):
    def __init__(self, work: Work):
        super(Scheduler, self).__init__(work)

    def calc(self, work: Work):
        self.sgd(work)
        return
