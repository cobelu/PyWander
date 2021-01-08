import time
import logging
from typing import List
import numpy as np
import ray
from queue import Queue

from line_profiler import line_profiler
from scipy.sparse import csr_matrix, csc_matrix, load_npz, coo_matrix

from parameters import Parameters
from partition import Partition

import atexit

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


def main():
    file = "data/netflix.npz"
    rows, cols, normalizer = Scheduler.load_dims(file)
    p = Parameters(
        sync=True,
        n=1,
        d=1,
        k=100,
        alpha=0.08,
        beta=0.05,
        lamda=1,
        normalizer=normalizer,
        file=file
    )
    scheduler = Scheduler(0, p, Partition(0, rows))
    total, nnz = scheduler.sgd(Partition(0, cols))
    print("NNZ: {0}".format(nnz))
    print("RMSE: {0}".format(np.sqrt(total / nnz)))


class Scheduler(object):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    def __init__(self, i: int, p: Parameters, work: Partition):
        Scheduler.logger.debug("Got file:" + p.file)
        self.i = i
        self.p = p
        self.queue = Queue()
        # Sparse data is loaded as a CSR, sliced, converted to CSC, and shuffled
        self.normalizer = p.normalizer
        self.a_csc: csc_matrix = self.load(p.file)[work.low:work.high].tocsc()
        Scheduler.logger.debug("Loading {0}".format(work))
        # Scheduler.logger.debug("Converted CSR to CSC")
        shape = self.a_csc.shape
        rows: int = shape[0]
        cols: int = shape[1]
        # Scheduler.logger.debug("Size of CSC: ({0}, {1})".format(shape[0], shape[1]))
        # self.a_csc = shuffle(self.a_csc)  # TODO: Shuffle
        # Data to be found
        rng = np.random
        self.w: np.ndarray = (1 / np.sqrt(self.p.k)) * rng.random((rows, self.p.k))
        self.h: np.ndarray = (1 / np.sqrt(self.p.k)) * np.asfortranarray(rng.random((self.p.k, cols)))
        self.tmp: np.ndarray = np.zeros(self.p.k)

    @profile
    def sgd(self, work: Partition) -> (float, int):
        # work = self.queue.get()
        # TODO: Using CPU time? Check in on time.time()? Want wall clock time
        # Scheduler.logger.debug("Crunching on ({0}, {1})".format(work.low, work.high))
        # Keeping track of RMSE along the way
        now = time.time()
        nnz_ctr = 0
        total = 0
        # Mark the low and high
        for j in range(work.low, work.high):
            hj = self.h[:, j]  # TODO: Nice syntax might be hiding performance
            for i_iter in range(self.a_csc.indptr[j], self.a_csc.indptr[j + 1]):
                i = self.a_csc.indices[i_iter]
                # Get the respective entries
                wi = self.w[i]
                aij = self.a_csc.data[i_iter]
                # Error = [(Wi • Hj) - Aij]
                err = aij - np.dot(wi, hj)
                # Descent
                np.copyto(self.tmp, wi)  # Temp stored for wi to be replaced gracefully
                # Wi -= lrate * (err*Hj + lambda*Wi)
                self.w[i] -= self.p.alpha * (err * hj + self.p.lamda * wi)
                # Hj -= lrate * (err*tmp + lambda*Hj);
                self.h[:, j] -= self.p.alpha * (err * self.tmp + self.p.lamda * hj)
                # Note the count of the nnz
                nnz_ctr += self.p.k
            # # Calculate RMSE
            # test_wi = wi * np.sqrt(self.normalizer)
            # test_hj = hj * np.sqrt(self.normalizer)
            # err = aij - np.dot(test_wi, test_hj)  # (yi' - yi)
            # term = np.power(err, 2)  # (yi' - yi)^2
            # total += term  # Σ_{i=1}^{n} (yi' - yi)^2
        # self.send(work)
        # a: coo_matrix = self.a_csc[:, work.low:work.high].tocoo()
        # for i, j, aij in zip(a.row, a.col, a.data):
        #     # Get the respective entries
        #     wi: np.ndarray = self.w[i]
        #     hj: np.ndarray = self.h[:, j]
        #     # aij = a[i, j]
        #     # Error = [(Wi • Hj) - Aij]
        #     err = aij - np.dot(wi, hj)
        #     np.copyto(self.tmp, wi)  # Temp stored for wi to be replaced gracefully
        #     # Descent
        #     # Wi -= lrate * (err*Hj + lambda*Wi)
        #     wi -= self.p.alpha * (err * hj + self.p.lamda * wi)
        #     # Hj -= lrate * (err*tmp + lambda*Hj);
        #     hj -= self.p.alpha * (err * self.tmp + self.p.lamda * hj)
        #     # Calculate RMSE
        #     test_wi = wi * np.sqrt(self.normalizer)
        #     test_hj = hj * np.sqrt(self.normalizer)
        #     err = aij - np.dot(test_wi, test_hj)  # (yi' - yi)
        #     term = err ** 2  # (yi' - yi)^2
        #     total += term  # Σ_{i=1}^{n} (yi' - yi)^2
        #     # Note the count of the nnz
        #     nnz_ctr += 1
        # TODO: Return vector instead
        # id = ray.put(self.h)
        print("Done in {0}s".format(time.time() - now))
        return total, nnz_ctr

    def load(self, filename: str) -> csr_matrix:
        # Scheduler.logger.debug("Loading " + filename)
        try:
            sparse_matrix: csr_matrix = load_npz(filename)
            # Normalize per: https://stackoverflow.com/a/62690439
            sparse_matrix /= self.normalizer
            Scheduler.logger.debug("Loaded {0}".format(filename))
        except IOError:
            Scheduler.logger.debug("Could not find file!")
            raise Exception("oops")
        return sparse_matrix

    def send(self, works: List[Partition]):
        workers = self.p.n
        another_worker = (self.i + 1 + np.random.randint(workers - 1)) % workers
        # return self.workers[another_worker].dump(work)
        for work in works:
            self.queues[another_worker].put(work)

    def dump(self, works: List[Partition]) -> bool:
        for work in works:
            self.queue.put(work)
        return True

    def ready(self) -> bool:
        return True

    @staticmethod
    def load_dims(filename: str) -> (int, int, float):
        Scheduler.logger.debug("Getting dimensions of {0}".format(filename))
        try:
            sparse_matrix: csr_matrix = load_npz(filename)
            shape = sparse_matrix.shape
            normalizer = sparse_matrix.max()
        except IOError:
            Scheduler.logger.warning("Could not find file!")
            shape = (0, 0)
            normalizer = 1
        return shape[0], shape[1], normalizer


if __name__ == '__main__':
    main()
