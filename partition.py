from random import shuffle
from typing import Type
from scipy.sparse import csr_matrix

class Partition:
    def __init__(self, low: int, high: int, h, prev):
        assert low < high, "Low should be less than high"
        self.low = low
        self.high = high
        self.h = h
        self.prev = prev

    def __str__(self):
        return "Partition{" + str(self.low) + ", " + str(self.high) + "}"
