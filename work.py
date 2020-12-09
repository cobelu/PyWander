from random import shuffle
from typing import Type
from scipy.sparse import csr_matrix


class Work:
    def __init__(self, low: int, high: int):
        assert low <= high, "Low should be greater than high"
        self.low = low
        self.high = high

    @classmethod
    def from_csr(cls, csr: csr_matrix):
        low = 0  # Beginning is 0
        high = csr.shape()[0]  # End is the number of rows
        return cls(low, high)

    def splits(self, n: int, shuffled=False):
        dim = self.dim()
        avg = dim // n
        extra = self.dim() % n
        # Add everything that's average to the list
        works = []
        last = self.low
        for i in range(n-extra):
            nxt = last + avg
            works.append(Work(last, nxt))
            last = nxt
        # A few will be "extra", so they are added second
        for i in range(n-extra, n):
            nxt = last + avg + 1
            works.append(Work(last, nxt))
            last = nxt
        if shuffled:
            shuffle(works)
        return works

    def individuals(self):
        n = self.dim()
        return [Work(self.low + i, self.low + (i + 1)) for i in range(n)]

    def dim(self):
        return self.high - self.low

    def __str__(self):
        return "Work{" + str(self.low) + ", " + str(self.high) + "}"
