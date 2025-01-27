import profile

import line_profiler
import time
import timeit
import logging
from typing import List, Any
import numpy as np
import ray
from ray.util.queue import Queue
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.utils import shuffle

from parameters import Parameters
from partition import Partition
from util import load
from work import Work


@ray.remote
class Worker(object):
    rng = np.random.default_rng()
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    def __init__(self, worker_id: int, p: Parameters, pending: Queue, complete: Queue, results: Queue,
                 a_csc: csc_matrix, row_partition: Partition):
        if p.verbose:
            Worker.logger.setLevel(logging.DEBUG)
        Worker.logger.debug("Got partition: {0}".format(row_partition))

        # Args
        self.worker_id = worker_id
        self.p = p
        self.pending = pending
        self.complete = complete
        self.results = results
        self.a_csc = a_csc
        # self.a_csc = shuffle(self.a_csc)  # TODO: Shuffle
        # Get the shape to know how large the parameter matrices
        shape = self.a_csc.shape
        rows: int = shape[0]  # Number of rows ON THE MACHINE
        # cols: int = shape[1]
        Worker.logger.debug("Size of CSC: ({0}, {1})".format(shape[0], shape[1]))
        # Create the arrays for computation
        self.w: np.ndarray = 1 / np.sqrt(self.p.k) * Worker.random((rows, self.p.k))
        self.tmp: np.ndarray = np.empty(self.p.k, dtype=np.float64)  # Pre-allocated
        # h is assigned later

    @ray.method(num_returns=0)
    def run(self):
        while True:
            work = self.pending.get()
            if self.p.n > 1 and work.prev == self.worker_id:
                self.pending.put(work)
                continue
            work, nnz, total = self.sgd(work)
            self.results.put((nnz, total))
            if self.p.sync:
                self.complete.put(work)
            else:
                self.pending.put(work)
        return

    @ray.method(num_returns=3)
    def sgd(self, work: Work):
        # start = time.time()
        Worker.logger.debug("Crunching on {0}".format(work))
        total = 0.0
        nnz = 0
        h = np.copy(work.h)
        # if self.p.bold:
        #     step = self.step_size(work)
        # else:
        #     step = 1
        # print("Work dim: {0}".format(work.dim()))
        work_dim = work.dim()
        # TODO: Indexing over h or a_csc??? FIX THIS ASAP
        for j in range(work.low(), work.high()):
            col = j - work.low()
            hj = h[:, col]
            # print("Length: {0}".format(len(self.a_csc.indptr[j:j + 1])))
            # print("From: {0}; To: {1}".format(self.a_csc.indptr[j], self.a_csc.indptr[j + 1]))
            for i_iter in range(self.a_csc.indptr[j], self.a_csc.indptr[j + 1]):
                # TODO: NOT EXECUTING HERE FOR NOMAD
                i = self.a_csc.indices[i_iter]
                # Get the W row
                wi = self.w[i]
                # Get the respective entry
                aij = self.a_csc.data[i_iter]
                # Error = [(Wi • Hj) - Aij]
                err = np.dot(wi, hj) - aij
                # Get the temp vector for calculation
                np.copyto(self.tmp, wi)  # Temp stored for wi to be replaced gracefully
                # Descent
                # Wi -= lrate * (err*Hj + lambda*Wi)
                wi -= self.p.alpha * (err * hj + self.p.lamda * wi)  # * step
                # Hj -= lrate * (err*tmp + lambda*Hj);
                hj -= self.p.alpha * (err * self.tmp + self.p.lamda * hj)  # * step
                # Calculate RMSE
                total += np.power(err, 2)  # Σ_{i=1}^{n} (yi' - yi)^2
                # Note the count of the nnz
                nnz += 1
        # stop = time.time()
        # Worker.logger.debug("Worker {0} done in {1}".format(self.worker_id, stop - start))
        # print("Returning NNZ: {0} & total: {1}".format(nnz, total))
        return Work(work.ptn, h, self.worker_id, work.updates + 1), nnz, total

    def step_size(self, work: Work):
        return self.p.lamda * 1.5 / (1.0 + self.p.beta * pow(work.updates + 1, 1.5))

    def lr(self, alpha: float):
        Worker.logger.debug("Setting alpha on {0}".format(self.worker_id))
        self.p.alpha = alpha

    @staticmethod
    def random(shape: (int, int)) -> np.ndarray:
        return Worker.rng.random(shape, dtype=np.float64)
