import ray
import numpy as np

from scheduler import Scheduler
from work import Work


@ray.remote
class Sync(Scheduler):
    def __init__(self, file: str, normalizer: float, work: Work):
        Scheduler.__init__(self, file, normalizer, work)
