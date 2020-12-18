import profile

import line_profiler
import time
import timeit
import logging
from typing import List, Any
import numpy as np
import ray
from ray.util.queue import Queue
from scipy.sparse import csr_matrix, csc_matrix, load_npz
from sklearn.utils import shuffle

from parameters import Parameters
from partition import Partition
from work import Work


@ray.remote
class Worker(object):
    rng = np.random.default_rng()
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    def __init__(self, worker_id: int, p: Parameters, pending: Queue, complete: Queue, row_partition: Partition):
        Worker.logger.debug("Got file:" + p.file)
        # Args
        self.worker_id = worker_id
        self.p = p
        self.pending = pending
        self.complete = complete
        # # Keeping track of RMSE along the way
        self.total = 0
        self.nnz = 0
        # Sparse data is loaded as a CSR, sliced, converted to CSC, and shuffled (if desired)
        self.normalizer = p.normalizer
        self.a_csc: csc_matrix = self.load(p.file)[row_partition.low:row_partition.high].tocsc()
        # self.a_csc = shuffle(self.a_csc)  # TODO: Shuffle
        # Get the shape to know how large the parameter matrices
        shape = self.a_csc.shape
        rows: int = shape[0]  # Number of rows ON THE MACHINE
        cols: int = shape[1]
        Worker.logger.debug("Size of CSC: ({0}, {1})".format(shape[0], shape[1]))
        # Create the arrays for computation
        self.w: np.ndarray = 1 / np.sqrt(self.p.k) * self.random((rows, self.p.k))
        self.tmp: np.ndarray = np.zeros(self.p.k, dtype=np.float64)  # Pre-allocated
        # Data to be found
        col_range = cols // self.p.n
        part_range = col_range // self.p.ptns
        for i in range(col_range // part_range):
            low = col_range * self.worker_id + part_range * i
            high = low + part_range
            h: np.ndarray = 1 / np.sqrt(self.p.k) * self.random((self.p.k, cols))
            self.complete.put(Work.initialize(low, high, h, -1))

    @ray.method(num_returns=2)
    def update(self):
        return self.total, self.nnz

    @ray.method(num_returns=0)
    def run(self):
        while True:
            work = self.pending.get()
            if work is None:
                print("done")
                break
            if self.p.n > 1 and work.prev == self.worker_id:
                self.pending.put(work)
                continue
            self.sgd(work)  # We modify the work inside the function
            work.prev = self.worker_id  # Note the last worker
            if self.p.sync:
                self.complete.put(work)
            else:
                self.pending.put(work)
        return

    @ray.method(num_returns=0)
    def sgd(self, work: Work):
        # TODO: Using CPU time? Check in on time.time()? Want wall clock time
        start = time.time()
        Worker.logger.debug("Crunching on {0}".format(work))
        # Scheduler.logger.debug("Crunching on ({0}, {1})".format(work.low, work.high))
        # Mark the low and high
        for j in range(work.dim()):
            hj = work.h[:, j]  # TODO: Nice syntax might be hiding performance
            for i_iter in range(self.a_csc.indptr[j], self.a_csc.indptr[j + 1]):
                i = self.a_csc.indices[i_iter]
                # Get the W row
                wi = self.w[i]
                # Get the respective entry
                aij = self.a_csc.data[i_iter]
                # Error = [(Wi • Hj) - Aij]
                err = aij - np.dot(wi, hj)
                np.copyto(self.tmp, wi)  # Temp stored for wi to be replaced gracefully
                # Descent
                # Wi -= lrate * (err*Hj + lambda*Wi)
                wi -= self.p.alpha * (err * hj + self.p.lamda * wi)
                # Hj -= lrate * (err*tmp + lambda*Hj);
                hj -= self.p.alpha * (err * self.tmp + self.p.lamda * hj)
                # Calculate RMSE
                # test_wi = wi * np.sqrt(self.normalizer)
                # test_hj = hj * np.sqrt(self.normalizer)
                # err = aij - np.dot(test_wi, test_hj)  # (yi' - yi)
                term = np.power(err, 2)  # (yi' - yi)^2
                self.total += term  # Σ_{i=1}^{n} (yi' - yi)^2
                # Note the count of the nnz
                self.nnz_ctr += 1
        stop = time.time()
        Worker.logger.debug("Worker {0} done in {1}".format(self.worker_id, stop - start))
        return

    def load(self, filename: str) -> csr_matrix:
        Worker.logger.debug("Loading " + filename)
        try:
            sparse_matrix: csr_matrix = load_npz(filename)
            # Normalize per: https://stackoverflow.com/a/62690439
            sparse_matrix /= self.normalizer
            Worker.logger.debug("Loaded {0}".format(filename))
        except IOError:
            Worker.logger.debug("Could not find file!")
            raise Exception("oops")
        return sparse_matrix

    def ready(self) -> int:
        return self.worker_id

    def random(self, shape: (int, int)) -> np.ndarray:
        return Worker.rng.random(shape, dtype=np.float64)

    @staticmethod
    def load_dims(filename: str) -> (int, int, int, float):
        Worker.logger.debug("Getting dimensions of {0}".format(filename))
        try:
            sparse_matrix: csr_matrix = load_npz(filename)
            shape = sparse_matrix.shape
            normalizer = sparse_matrix.max()
            nnz = sparse_matrix.nnz
        except IOError:
            Worker.logger.warning("Could not find file!")
            shape = (-1, -1)
            nnz = -1
            normalizer = -1
        return shape[0], shape[1], nnz, normalizer
