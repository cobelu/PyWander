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
        try:
            self.a_csc: csr_matrix = load_npz(file)
            Scheduler.logger.debug("Loaded CSR file: " + file)
        except IOError:
            Scheduler.logger.debug("Could not find file " + file)
        self.a_csc: csc_matrix = self.a_csc[work.low:work.high].tocsc()
        Scheduler.logger.debug("Converted CSR to CSC")
        self.a_csc = shuffle(self.a_csc)
        # Data to be found
        self.w = np.random.rand(self.k) * 1.0 / np.sqrt(self.k)
        self.h = np.random.rand(self.k) * 1.0 / np.sqrt(self.k)

    def sgd(self, work: Work, h: np.ndarray):
        Scheduler.logger.debug("Working on (" + str(work.low) + ", " + str(work.high) + ") on " + str(h))
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
        nz_ids = zip(nz_ids[0], nz_ids[1])
        try:
            for i, j in nz_ids:
                Scheduler.logger.debug("Considering: (" + str(i) + ", " + str(j) + ")")
                # Get the respective entries
                aij = self.a_csc[i][j]
                wi = self.w[i]
                hj = self.h[j]
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
    def load(filename: str):
        Scheduler.logger.debug("Loading " + filename)
        try:
            sparse_matrix = load_npz(filename)
            Scheduler.logger.debug("Loaded " + filename)
        except IOError:
            Scheduler.logger.debug("Could not find file!")
            raise Exception("oops")
        return sparse_matrix
