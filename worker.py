import logging
import numpy as np
import ray
import time
import timeit
from ray.util.queue import Queue
from scipy.sparse import csr_matrix, csc_matrix, load_npz
from typing import List, Any

from partition import Partition

@ray.remote
class Worker(object):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    def __init__(self, worker_id: int, args, pending: Queue, complete: Queue):
        self.worker_id = worker_id
        self.args = args
        self.pending = pending
        self.complete = complete
        self.total = 0
        self.nnz = 0
        self.tmp: np.ndarray = np.zeros(self.args.latent)

        Worker.logger.debug('Loading file: ' + self.args.filename + '...')
        a = self.load(args.filename)
        shape = a.shape
        rows: int = shape[0]
        cols: int = shape[1]

        row_range = rows // self.args.workers
        self.low = row_range * self.worker_id
        self.high = self.low + row_range if self.low + row_range < rows else rows

        self.a_csc: csc_matrix = a[self.low:self.high].tocsc()
        rng = np.random.default_rng()
        self.w: np.ndarray = ((1 / np.sqrt(self.args.latent))
                               * rng.random((rows, self.args.latent)))

        col_range = cols // self.args.workers
        part_range = col_range // self.args.partitions
        for i in range(col_range // part_range):
            low = col_range * self.worker_id + part_range * i
            high = low + part_range
            h: np.ndarray = ((1 / np.sqrt(self.args.latent))
                              * rng.random((self.args.latent, cols)))
            #print("low, high=", low, high)
            self.complete.put(Partition(low, high, h, -1))

    @ray.method(num_returns=2)
    def update(self):
        return self.total, self.nnz

    @ray.method(num_returns=0)
    def run(self):
        while True:
            part = self.pending.get()

            if part is None:
                print("done")
                break

            if self.args.workers > 1 and part.prev == self.worker_id:
                self.pending.put(part)
                continue

            h = self.sgd(part)
            part = Partition(part.low, part.high, h, self.worker_id)
            if self.args.sync:
                self.complete.put(part)
            else:
                self.pending.put(part)

    def sgd(self, part):
        start = time.time() #check timer
        h = np.copy(part.h)
        for j in range(part.low, part.high):
            hj = h[:, j]
            for i_iter in range(self.a_csc.indptr[j], self.a_csc.indptr[j + 1]):
                i = self.a_csc.indices[i_iter]
                # Get the respective entries
                wi: np.ndarray = self.w[i]
                aij = self.a_csc.data[i_iter]
                # Error = [(Wi • Hj) - Aij]
                err = aij - np.dot(wi, hj)
                np.copyto(self.tmp, wi)  # Temp stored for wi to be replaced gracefully
                # Descent
                # Wi -= lrate * (err*Hj + lambda*Wi)
                wi -= self.args.alpha * (err * hj + self.args.lamda * wi)
                # Hj -= lrate * (err*tmp + lambda*Hj);
                hj -= self.args.alpha * (err * self.tmp + self.args.lamda * hj)
                # Calculate RMSE
                # test_wi = wi * np.sqrt(self.normalizer)
                # test_hj = hj * np.sqrt(self.normalizer)
                # err = aij - np.dot(test_wi, test_hj)  # (yi' - yi)
                # term = np.power(err, 2)  # (yi' - yi)^2
                # total += term  # Σ_{i=1}^{n} (yi' - yi)^2
                # Note the count of the nnz
                self.nnz += 1
        stop = time.time()
        Worker.logger.debug("Worker {0} done in {1}".format(self.worker_id, stop - start))
        return h

    def load(self, filename: str) -> csr_matrix:
        Worker.logger.debug("Loading " + filename)
        try:
            sparse_matrix: csr_matrix = load_npz(filename)
            # Normalize per: https://stackoverflow.com/a/62690439
            #sparse_matrix /= self.normalizer
            Worker.logger.debug("Loaded {0}".format(filename))
        except IOError:
            Worker.logger.debug("Could not find file!")
            raise Exception("oops")
        return sparse_matrix
