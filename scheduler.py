import logging
from typing import List, Any
import numpy as np
import ray
from queue import Queue
from scipy.sparse import csr_matrix, csc_matrix, load_npz
from sklearn.utils import shuffle

from parameters import Parameters
from work import Work


@ray.remote
class Scheduler(object):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    def __init__(self, i: int, p: Parameters, work: Work):
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
        rng = np.random.default_rng()
        self.w: np.ndarray = (1 / np.sqrt(self.p.k)) * rng.random((rows, self.p.k))
        self.h: np.ndarray = (1 / np.sqrt(self.p.k)) * rng.random((self.p.k, cols))

    @ray.method(num_returns=3)
    def sgd(self, h=None) -> (float, int, np.ndarray):
        work = self.queue.get()
        Scheduler.logger.debug("Crunching on {0}".format(work))
        # Scheduler.logger.debug("Crunching on ({0}, {1})".format(work.low, work.high))
        if h is not None:
            self.h = np.asarray(np.copy(h))  # Ray objects are immutable
        # Keeping track of RMSE along the way
        nnz_ctr = 0
        total = 0
        offset = work.low
        # Mark the low and high
        low = work.low - offset
        high = work.high - offset
        for j in range(low, high):
            hj = self.h[:, j]
            for i_iter in range(self.a_csc.indptr[j], self.a_csc.indptr[j + 1]):
                i = self.a_csc.indices[i_iter]
                # Get the respective entries
                wi = self.w[i]
                aij = self.a_csc.data[i_iter]
                # Error = [(Wi • Hj) - Aij]
                err = aij - np.dot(wi, hj)
                tmp = wi  # Temp stored for wi to be replaced gracefully
                # Descent
                # Wi -= lrate * (err*Hj + lambda*Wi)
                wi -= self.p.alpha * (err * hj + self.p.lamda * wi)
                # Hj -= lrate * (err*tmp + lambda*Hj);
                hj -= self.p.alpha * (err * tmp + self.p.lamda * hj)
                # Calculate RMSE
                test_wi = wi * np.sqrt(self.normalizer)
                test_hj = hj * np.sqrt(self.normalizer)
                err = aij - np.dot(test_wi, test_hj)  # (yi' - yi)
                term = np.power(err, 2)  # (yi' - yi)^2
                total += term  # Σ_{i=1}^{n} (yi' - yi)^2
                # Note the count of the nnz
                nnz_ctr += 1
        self.send(work)
        return total, nnz_ctr, self.h

    def load(self, filename: str) -> csr_matrix:
        Scheduler.logger.debug("Loading " + filename)
        try:
            sparse_matrix: csr_matrix = load_npz(filename)
            # Normalize per: https://stackoverflow.com/a/62690439
            sparse_matrix /= self.normalizer
            Scheduler.logger.debug("Loaded {0}".format(filename))
        except IOError:
            Scheduler.logger.debug("Could not find file!")
            raise Exception("oops")
        return sparse_matrix

    def send(self, works: List[Work]):
        workers = self.p.n
        another_worker = (self.i + 1 + np.random.randint(workers - 1)) % workers
        # return self.workers[another_worker].dump(work)
        for work in works:
            self.queues[another_worker].put(work)

    def dump(self, works: List[Work]) -> bool:
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
