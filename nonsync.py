from typing import Type
import numpy as np
import ray

from scheduler import Scheduler
from work import Work


@ray.remote
class Async(Scheduler):
    def __init__(self, work: Work):
        super(Scheduler, self).__init__(work)

    async def calc(self, work: Work, h: np.ndarray) -> np.ndarray:
        self.sgd(work)
        return
