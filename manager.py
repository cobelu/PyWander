from typing import List

import numpy as np
import ray
import time
from ray.util.queue import Queue
from scipy.sparse import csr_matrix, load_npz

from parameters import Parameters
from partition import Partition
from util import load, load_with_features
from worker import Worker


class Manager:
    def __init__(self, p: Parameters):
        self.p = p
        self.nnz = 0
        self.rmse = 0
        self.pending = Queue()
        self.complete = Queue()
        self.results = Queue()

        self.num_col_parts = self.p.n * self.p.ptns
        a_csr = load(self.p.filename, self.p.normalize)
        row_ptns = Partition(0, a_csr.shape[0]).splits(self.p.n, False)

        self.workers = []
        for i in range(self.p.n):
            row_partition = row_ptns[i]
            a_csc = a_csr[row_partition.low:row_partition.high].tocsc()
            self.workers.append(Worker.remote(i, p, self.pending, self.complete,
                                self.results, a_csc, row_partition))
        [worker.run.remote() for worker in self.workers]

    def run(self):
        raise NotImplementedError()

    def print_rmse(self):
        print("\tNNZ: {0}".format(self.nnz))
        rmse = np.sqrt(self.rmse) if self.nnz > 0 else np.NaN
        print("\tRMSE: {0}".format(rmse))

    def shutdown(self):
        [ray.kill(worker) for worker in self.workers]


class SyncManager(Manager):
    def run(self):
        start = time.time()
        for step in range(1, self.p.d + 1):
            print("Iteration: {0}".format(step))
            # Place initial work on queue
            for i in range(self.num_col_parts):
                self.pending.put(self.complete.get())
            for i in range(self.num_col_parts):
                nnz, total = self.results.get()
                #https://stats.stackexchange.com/questions/221826/is-it-possible-to-compute-rmse-iteratively
                #double check this
                self.nnz += nnz
                self.rmse = ((self.nnz - nnz) / self.nnz) * self.rmse + total / self.nnz
            if step % self.p.report == 0:
                self.print_rmse()
        print('FINAL')
        self.print_rmse()
        print('runtime:', time.time() - start)
        self.shutdown()


class AsyncManager(Manager):
    def run(self):
        start = time.time()
        for i in range(self.num_col_parts):
            self.pending.put(self.complete.get())
        while True:
            while not self.results.empty():
                nnz, total = self.results.get()
                #https://stats.stackexchange.com/questions/221826/is-it-possible-to-compute-rmse-iteratively
                #double check this
                self.nnz += nnz
                self.rmse = ((self.nnz - nnz) / self.nnz) * self.rmse + total / self.nnz
            time.sleep(self.p.report)
            elapsed = time.time() - start
            if elapsed > self.p.d:
                break
            print('Elapsed:', elapsed)
            self.print_rmse()
        print('FINAL')
        self.print_rmse()
        print('runtime:', time.time() - start)
        self.shutdown()
