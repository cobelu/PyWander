from typing import Type
import numpy as np
import ray
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, load_npz

from work import Work


@ray.remote
class Scheduler(object):

    def __init__(self, work: Work):
        # Parameters
        self.k = 100
        self.alpha = 0.08
        self.beta = 0.05
        self.lamda = 1
        self.file = ""
        # Sparse data is loaded as a CSR, sliced, and converted to CSC
        self.a_csc: csc_matrix = load(self.file)[work.low:work.high].tocsc()
        # Data to be found
        self.w = np.rand.random(self.k) * 1.0/np.sqrt(self.k)
        self.h = np.rand.random(self.k) * 1.0/np.sqrt(self.k)

    def sgd(self, work: Work):
        # Keeping track of RMSE along the way
        nnz_ctr = 0
        sum = 0
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
            sum += term  # Σ_i=1 ^ n(yi' - yi)^2
        nnz_ctr += nnz


def load(filename: str):
    # TODO: This load method
    try:
        sparse_matrix = load_npz(filename)
    except IOError:
        print("Could not find file!")
        quit()
    return sparse_matrix
