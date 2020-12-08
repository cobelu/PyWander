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

    def __init__(self, file: str, work: Work):
        Scheduler.logger.debug("Got file:" + file)
        # Parameters
        self.k = 100
        self.alpha = 0.08
        self.beta = 0.05
        self.lamda = 1
        # Sparse data is loaded as a CSR, sliced, converted to CSC, and shuffled
        self.a_csc: csc_matrix = Scheduler.load(file)[work.low:work.high].tocsc()
        Scheduler.logger.debug("Loading ({0}, {1})".format(work.low, work.high))
        Scheduler.logger.debug("Converted CSR to CSC")
        shape = self.a_csc.shape
        Scheduler.logger.debug("Size of CSC: ({0}, {1})".format(shape[0], shape[1]))
        # self.a_csc = shuffle(self.a_csc)  # TODO: Shuffle
        # Data to be found
        self.w = np.random.rand(self.k) * 1.0 / np.sqrt(self.k)
        self.h = np.random.rand(self.k) * 1.0 / np.sqrt(self.k)

    def sgd(self, work: Work, h: np.ndarray):
        Scheduler.logger.debug("Working on ({0}, {1})".format(work.low, work.high))
        if h:
            self.h = h
        # Keeping track of RMSE along the way
        nnz_ctr = 0
        total = 0
        offset = work.low
        # Mark the low and high
        low = work.low - offset
        high = work.high - offset
        nnz = self.a_csc[low:high, :].getnnz()
        nz_ids = self.a_csc.nonzero()
        Scheduler.logger.debug("NNZ: {0}".format(len(nz_ids[0])))
        try:
            for j in range(low, high):
                hj = self.h[j]
                for i_iter in range(self.a_csc.indptr[j], self.a_csc.indptr[j+1]):
                    i = self.a_csc.indices[i_iter]
                    # Get the respective entries
                    aij = self.a_csc.data[i_iter]
                    Scheduler.logger.debug("Considering ({0}, {1}): {2}".format(i, j, aij))
                    wi = self.w[i]
                    # error = [(Wi • Hj) - Aij]
                    err = aij - np.dot(wi, hj)
                    tmp = wi
                    # Wi -= lrate * (err*Hj + lambda*Wi)
                    wi -= self.alpha * (err*hj + self.lamda*wi)
                    # Hj -= lrate * (err*tmp + lambda*Hj);
                    hj -= self.alpha * (err*tmp + self.lamda*hj)
                    # Calculate Error
                    entry = np.Dot(wi, hj)  # yi'
                    term = np.power(entry - aij, 2)  # (yi' - yi)^2
                    total += term  # Σ_i=1 ^ n(yi' - yi)^2
                    nnz += 1
        except TimeoutException:
            return total, nnz_ctr, self.h
        return total, nnz_ctr, self.h

    @staticmethod
    def load(filename: str) -> csr_matrix:
        Scheduler.logger.debug("Loading " + filename)
        try:
            sparse_matrix: csr_matrix = load_npz(filename)
            Scheduler.logger.debug("Loaded {0}".format(filename))
        except IOError:
            Scheduler.logger.debug("Could not find file!")
            raise Exception("oops")
        return sparse_matrix

    @staticmethod
    def load_dims(filename: str) -> (int, int):
        Scheduler.logger.debug("Getting dimensions of {0}".format(filename))
        try:
            sparse_matrix: csr_matrix = load_npz(filename)
            shape = sparse_matrix.shape
        except IOError:
            Scheduler.logger.warning("Could not find file!")
            shape = (0, 0)
        return shape
