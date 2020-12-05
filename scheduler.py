import logging
from typing import Type
import numpy as np
import ray
from scipy.sparse import csr_matrix, csc_matrix, load_npz
from sklearn.utils import shuffle

from work import Work


class Scheduler(object):

    def __init__(self, file: str, work: Work):
        # Verify message was passed correctly
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Got file" + file)
        # Parameters
        self.k = 100
        self.alpha = 0.08
        self.beta = 0.05
        self.lamda = 1
        # Sparse data is loaded as a CSR, sliced, converted to CSC, and shuffled
        self.a_csc: csc_matrix = load(file)[work.low:work.high].tocsc()
        self.a_csc = shuffle(self.a_csc)
        self.logger.debug("Loaded CSC")
        # Data to be found
        self.w = np.rand.random(self.k) * 1.0 / np.sqrt(self.k)
        self.h = np.rand.random(self.k) * 1.0 / np.sqrt(self.k)

    def sgd(self, work: Work, h: np.ndarray):
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
        for i, j in nz_ids:
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
        nnz_ctr += nnz
        return total, nnz_ctr, self.h


def load(filename: str):
    try:
        sparse_matrix = load_npz(filename)
    except IOError:
        sparse_matrix = csr_matrix(np.array(0))
        print("Could not find file!")
        quit()
    return sparse_matrix
