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

    def ptn_dsgd(self, n: int, shuffled=False) -> List[Partition]:
        dim = self.dim()
        # "Evenly" split blocks
        avg = dim // n
        extra = dim % n
        return self.ptn_helper(n, avg, extra, shuffled)

    def ptn_dsgdpp(self, n: int, shuffled=False) -> List[Partition]:
        dim = self.dim()
        # Block sizes are halved
        avg = (dim // n) // 2
        extra = dim % (n // 2)
        return self.ptn_helper(n, avg, extra, shuffled)

    def ptn_fpsgd(self, n: int, shuffled=False) -> List[Partition]:
        dim = self.dim()
        # "Evenly" split blocks
        avg = dim // (n + 1)
        extra = dim % (n + 1)
        return self.ptn_helper(n, avg, extra, shuffled)

    def ptn_nomad(self) -> List[Partition]:
        n = self.dim()
        # TODO: Check indexing
        return [Partition(self.low + i, self.low + (i + 1)) for i in range(n)]

    def ptn_helper(self, n: int, avg: int, extra: int, shuffled=False):
        works: List[Partition] = []
        last = self.low
        # Add the averages to the list
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

    def dim(self) -> int:
        return self.high - self.low

    def __str__(self):
        return "Ptn{" + str(self.low) + ", " + str(self.high) + "}"
