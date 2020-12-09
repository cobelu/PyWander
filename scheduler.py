import logging
from typing import Type
import numpy as np
import ray
from scipy.sparse import csr_matrix, csc_matrix, load_npz
from sklearn.utils import shuffle

from timeout import TimeoutException
from work import Work


class Scheduler(object):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    def __init__(self, file: str, normalizer: float, work: Work):
        Scheduler.logger.debug("Got file:" + file)
        # Parameters
        self.k = 100
        self.alpha = 0.08
        self.beta = 0.05
        self.lamda = 1
        # Sparse data is loaded as a CSR, sliced, converted to CSC, and shuffled
        self.normalizer = normalizer
        self.a_csc: csc_matrix = self.load(file)[work.low:work.high].tocsc()
        Scheduler.logger.debug("Loading ({0}, {1})...".format(work.low, work.high))
        # Scheduler.logger.debug("Converted CSR to CSC")
        shape = self.a_csc.shape
        rows: int = shape[0]
        cols: int = shape[1]
        # Scheduler.logger.debug("Size of CSC: ({0}, {1})".format(shape[0], shape[1]))
        # self.a_csc = shuffle(self.a_csc)  # TODO: Shuffle
        # Data to be found
        rng = np.random.default_rng()
        self.w: np.ndarray = (1 / np.sqrt(self.k)) * rng.random((rows, self.k))
        print("w: (min: {0}, max: {1}, NaN: {2}, type: {3})".format(
            np.min(self.w), np.max(self.w), np.isnan(self.w).any(), self.w.dtype))
        self.h: np.ndarray = (1 / np.sqrt(self.k)) * rng.random((self.k, cols))
        print("h: (min: {0}, max: {1}, NaN: {2}, type: {3})".format(
            np.min(self.h), np.max(self.h), np.isnan(self.h).any(), self.h.dtype))

    def sgd(self, work: Work, h: np.ndarray):
        Scheduler.logger.debug("Crunching on ({0}, {1})".format(work.low, work.high))
        if h:
            self.h = h
        # Keeping track of RMSE along the way
        nnz_ctr = 0
        total = 0
        offset = work.low
        # Mark the low and high
        low = work.low - offset
        high = work.high - offset
        try:
            for j in range(low, high):
                hj = self.h[:, j]
                for i_iter in range(self.a_csc.indptr[j], self.a_csc.indptr[j+1]):
                    i = self.a_csc.indices[i_iter]
                    # Get the respective entries
                    wi = self.w[i]
                    aij = self.a_csc.data[i_iter]
                    # Error = [(Wi • Hj) - Aij]
                    err = aij - np.dot(wi, hj)
                    tmp = wi  # Temp stored for wi to be replaced gracefully
                    # Descent
                    # Wi -= lrate * (err*Hj + lambda*Wi)
                    wi -= self.alpha * (err*hj + self.lamda*wi)
                    # Hj -= lrate * (err*tmp + lambda*Hj);
                    hj -= self.alpha * (err*tmp + self.lamda*hj)
                    # Calculate RMSE
                    test_wi = wi * np.sqrt(self.normalizer)
                    test_hj = hj * np.sqrt(self.normalizer)
                    err = aij - np.dot(test_wi, test_hj)  # (yi' - yi)
                    term = np.power(err, 2)  # (yi' - yi)^2
                    total += term  # Σ_{i=1}^{n} (yi' - yi)^2
                    # Note the count of the nnz
                    nnz_ctr += 1
        except TimeoutException:
            return total, nnz_ctr, self.h
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

