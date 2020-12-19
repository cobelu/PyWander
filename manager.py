from typing import List

import numpy as np
import ray
import time
from ray.util.queue import Queue
from scipy.sparse import csr_matrix, load_npz

from parameters import Parameters
from partition import Partition
from worker import Worker


class Manager:
    def __init__(self, p: Parameters, filename: str):
        self.p = p
        self.total = 0
        self.nnz = 0
        self.pending = Queue()
        self.complete = Queue()
        self.results = Queue()

        self.num_col_parts = self.p.n * self.p.ptns
        a_csr, rows, cols, normalizer = self.load(filename)
        row_ptns = Partition(0, rows).splits(self.p.n, False)

        self.workers = [Worker.remote(i, p, self.pending, self.complete,
                                  self.results, a_csr, normalizer, row_ptns[i])
                        for i in range(self.p.n)]
        [worker.run.remote() for worker in self.workers]

    def run(self):
        raise NotImplementedError()

    def print_rmse(self):
        print("\tNNZ: {0}".format(self.nnz))
        rmse = np.sqrt(self.total / self.nnz) if self.nnz > 0 else np.NaN
        print("\tRMSE: {0}".format(rmse))

    def load(self, filename: str) -> csr_matrix:
        print("Loading " + filename)
        try:
            a_csr: csr_matrix = load_npz(filename)[0:10000]
            shape = a_csr.shape
            normalizer = a_csr.max()
            # Normalize per: https://stackoverflow.com/a/62690439
            a_csr /= normalizer
            print("Loaded {0}".format(filename))
        except IOError:
            print("Could not find file!")
            raise Exception("oops")
        return a_csr, shape[0], shape[1], normalizer

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
