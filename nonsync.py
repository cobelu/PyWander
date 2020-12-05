from typing import Type
import numpy as np
import ray

from scheduler import Scheduler
from work import Work


@ray.remote
class Async(Scheduler):
    def __init__(self, file: str, work: Work):
        Scheduler.__init__(self, file, work)

    async def calc(self, work: Work, h: np.ndarray) -> np.ndarray:
        self.sgd(work, h)
        return
