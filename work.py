import numpy as np

from partition import Partition


class Work:
    def __init__(self, ptn: Partition, h: np.ndarray, prev: int, updates: int):
        self.ptn: Partition = ptn
        self.h: np.ndarray = h
        self.prev: int = prev
        self.updates: int = updates

    def low(self) -> int:
        return self.ptn.low

    def high(self) -> int:
        return self.ptn.high

    def dim(self) -> int:
        return self.ptn.dim()

    @staticmethod
    def initialize(low: int, high: int, h: np.ndarray, prev: int):
        return Work(Partition(low, high), h, prev, 0)

    def __str__(self):
        return "Work({0})".format(self.ptn)
