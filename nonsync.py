import numpy as np
import ray

from scheduler import Scheduler
from work import Work


@ray.remote
class Async(Scheduler):
    def __init__(self, file: str, normalizer: float, work: Work):
        Scheduler.__init__(self, file, normalizer, work)
