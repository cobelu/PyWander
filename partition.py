from __future__ import annotations

from random import shuffle
from typing import Type, List
from scipy.sparse import csr_matrix
import numpy as np


class Partition:

    def __init__(self, low: int, high: int):
        assert low < high, "Low should be greater than high"
        self.low = low
        self.high = high

    @classmethod
    def from_csr(cls, csr: csr_matrix) -> Partition:
        low = 0  # Beginning is 0
        high = csr.shape()[0]  # End is the number of rows
        return cls(low, high)

    def splits(self, n: int, shuffled=False) -> List[Partition]:
        dim = self.dim()
        avg = dim // n
        extra = self.dim() % n
        # Add everything that's average to the list
        works = []
        last = self.low
        for _ in range(n - extra):
            nxt = last + avg
            works.append(Partition(last, nxt))
            last = nxt
        # A few will be "extra", so they are added second
        for _ in range(n - extra, n):
            nxt = last + avg + 1
            works.append(Partition(last, nxt))
            last = nxt
        if shuffled:
            shuffle(works)
        return works

    def individuals(self) -> List[Partition]:
        n = self.dim()
        return [Partition(self.low + i, self.low + (i + 1)) for i in range(n)]

    def dim(self) -> int:
        return self.high - self.low

    def __str__(self):
        return "Ptn{" + str(self.low) + ", " + str(self.high) + "}"