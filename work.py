from typing import List

import numpy as np
from ray.util.queue import Queue

from chunk import Chunk


class Work:
    def __init__(self, chunks: List[Chunk], hs: np.ndarray):
        self.chunks = chunks
        self.hs = hs

    @staticmethod
    def initial(self, i: int, k: int, cols: int, shuffled=False):
        # Create the initial chunk and split it
        chunks = Chunk(0, cols).splits(i, shuffled=shuffled)
        # Normalize
        coeff = 1 / np.sqrt(self.p.k)
        # Make enough h vectors for each Chunk
        hs = coeff * np.ndarray([Chunk.rng.random(
            (k, chunks[idx].dim())) for idx in range(i)])
        return [Work(chunks[idx], hs[idx]) for idx in range(i)]
