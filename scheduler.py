from typing import Type
import numpy as np
import ray
from scipy.sparse import coo_matrix, csr_matrix, load_npz


@ray.remote
class Scheduler(object):

    def __init__(self):
        # Parameters
        self.k = 100
        self.alpha = 0.08
        self.beta = 0.05
        self.lamda = 1
        self.file = ""
        # Sparse data
        self.a_coo: coo_matrix = load(self.file)
        self.a_csr: csr_matrix = self.a_coo.tocsr()
        # Data
        self.w = np.rand.random(self.k) * 1.0/np.sqrt(self.k)
        self.h = np.rand.random(self.k) * 1.0/np.sqrt(self.k)

    def sgd(self):
        for i, j, v in zip(self.a_coo.row, self.a_coo.col, self.a_coo.data):
            # Get the respective entries
            wi = self.w[i]
            hj = self.h[j]
            # error = [(Wi • Hj) - Aij]
            err = self.a_coo[i][j] - np.dot(wi, hj)
            tmp = wi
            # Wi -= lrate * (err*Hj + lambda*Wi)
            wi -= self.alpha * (err*hj + self.lamda*wi)
            # Hj -= lrate * (err*tmp + lambda*Hj);
            hj -= self.alpha * (err*tmp + self.lamda*hj)


def load(filename: str):
    # TODO: This load method
    try:
        sparse_matrix = load_npz(filename)
    except IOError:
        print("Could not find file!")
        quit()
    return sparse_matrix
