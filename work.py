from typing import Type


class Work:
    def __init__(self, low: int, high: int):
        assert low <= high, "Low should be greater than high"
        self.low = low
        self.high = high

    def splits(self, n: int):
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
            last = next
        return works

    def individuals(self):
        n = self.dim()
        return [Work(self.low + i, self.low + (i + 1)) for i in range(n)]

    def dim(self):
        return self.high - self.low
