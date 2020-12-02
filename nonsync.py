from typing import Type
import numpy as np
import ray

from scheduler import Scheduler


@ray.remote
class Async(Scheduler):
    def __init__(self):
        super(Scheduler, self).__init__()

    def read(self):
        return self.n

