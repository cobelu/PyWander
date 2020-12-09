from typing import List

import numpy as np
import ray
from ray.util.queue import Queue

from scheduler import Scheduler
from work import Work


@ray.remote
class Async(Scheduler):
    def __init__(self, i: int, file: str, normalizer: float, work: Work, queues: List[Queue]):
        Scheduler.__init__(self, file, normalizer, work, queues)
