import ray

from scheduler import Scheduler


@ray.remote
class Sync(Scheduler):
    def __init__(self):
        super(Scheduler, self).__init__()

    def read(self):
        return self.n