from typing import List

import numpy as np
from ray.util.queue import Queue

from partition import Partition


class Work:
    def __init__(self, ptn: Partition, h: np.ndarray, prev: int, updates: int):
        self.ptn = ptn
        self.h = h
        self.prev = prev
        self.updates = updates

    def low(self) -> int:
        return self.ptn.low

    def high(self) -> int:
        return self.ptn.high

    def dim(self) -> int:
        return self.ptn.dim()

    @staticmethod
    def initialize(low: int, high: int, h: np.ndarray, prev: int):
        return Work(Partition(low, high), h, prev)

    def __str__(self):
        return "Work{" + str(self.ptn) + ", " + str(self.prev) + "}"
