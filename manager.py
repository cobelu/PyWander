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
        self.total = 0
        self.nnz = 0
        self.pending = Queue()
        self.complete = Queue()
        self.results = Queue()

        self.num_col_parts = self.p.n * self.p.ptns
        _, rows, cols, normalizer = load_with_features(p.filename)
        row_ptns = Partition(0, rows).splits(self.p.n, False)

        self.workers = [Worker.remote(i, p, self.pending, self.complete,
                                  self.results, normalizer, row_ptns[i])
                        for i in range(self.p.n)]
        [worker.run.remote() for worker in self.workers]

    def run(self):
        raise NotImplementedError()

    def print_rmse(self):
        print("\tNNZ: {0}".format(self.nnz))
        rmse = np.sqrt(self.total / self.nnz) if self.nnz > 0 else np.NaN
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
                total, nnz = self.results.get()
                self.total += total
                self.nnz += nnz
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
            time.sleep(self.p.report)
            while not self.results.empty():
                total, nnz = self.results.get()
                self.total += total
                self.nnz += nnz
            elapsed = time.time() - start
            if elapsed > self.p.d:
                break
            print('Elapsed:', elapsed)
            self.print_rmse()
        print('FINAL')
        self.print_rmse()
        print('runtime:', time.time() - start)
        self.shutdown()
