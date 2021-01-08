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
from util import DSGD, DSGDPP, FPSGD, NOMAD


class Manager:
    def __init__(self, p: Parameters):
        self.p = p
        self.nnz = 0
        self.rmse = 0
        self.obj = 0
        self.pending = Queue()
        self.complete = Queue()
        self.results = Queue()

        self.num_col_parts = self.p.n * self.p.ptns
        a_csr = load(self.p.filename, self.p.normalize)
        rows, cols = a_csr.shape
        row_ptns = Partition(0, rows).ptn_dsgd(self.p.n, False)
        # Determine how to partition cols
        col_ptn = Partition(0, cols)
        if self.p.method == DSGDPP:
            col_ptns = col_ptn.ptn_dsgdpp(self.p.n)
        elif self.p.method == FPSGD:
            col_ptns = col_ptn.ptn_fpsgd(self.p.n)
        elif self.p.method == NOMAD:
            col_ptns = col_ptn.ptn_nomad()
        else:
            col_ptns = col_ptn.ptn_dsgd(self.p.n)

        self.workers = []
        for i in range(self.p.n):
            a_csc = a_csr[row_ptns[i].low:row_ptns[i].high].tocsc()
            self.workers.append(Worker.remote(i, p, self.pending, self.complete,
                                self.results, a_csc, row_ptns[i], col_ptns[i]))

        self.print_params()

    def run(self):
        raise NotImplementedError()

    def print_rmse(self, dur, step=None):
        rmse = np.sqrt(self.rmse) if self.nnz > 0 else np.NaN
        # print("\tNNZ: {0}".format(self.nnz))
        # print("\tRMSE: {0}".format(rmse))
        if step:
            print("{0},{1},{2},{3}".format(dur, self.nnz, rmse, step))
        else:
            print("{0},{1},{2}".format(dur, self.nnz, rmse))

    def print_params(self):
        print("sync,{0}".format(self.p.sync))
        print("n,{0}".format(self.p.n))
        print("d,{0}".format(self.p.d))
        print("k,{0}".format(self.p.k))
        print("alpha,{0}".format(self.p.alpha))
        print("beta,{0}".format(self.p.beta))
        print("lamda,{0}".format(self.p.lamda))
        print("ptns,{0}".format(self.p.ptns))
        print("report,{0}".format(self.p.report))
        print("normalize,{0}".format(self.p.normalize))
        print("bold,{0}".format(self.p.bold))
        print("method,{0}".format(self.p.method))
        print("-"*50)

    def shutdown(self):
        [ray.kill(worker) for worker in self.workers]


class SyncManager(Manager):
    def run(self):
        [worker.run.remote() for worker in self.workers]
        start = time.time()
        for step in range(1, self.p.d + 1):
            # print("Iteration: {0}".format(step))
            # Place initial work on queue
            for i in range(self.num_col_parts):
                self.pending.put(self.complete.get())
            for i in range(self.num_col_parts):
                nnz, total = self.results.get()
                # https://stats.stackexchange.com/questions/221826/is-it-possible-to-compute-rmse-iteratively
                # Double check this
                self.nnz += nnz
                self.rmse = ((self.nnz - nnz) / self.nnz) * self.rmse + total / self.nnz
                if self.p.bold:
                    obj = self.rmse + self.p.lamda * self.nnz  # TODO: Regularization term not necc'ly nnz
                    if obj > self.obj:
                        self.p.alpha *= self.p.beta
                    else:
                        self.p.alpha *= self.p.beta
                    [worker.lr.remote(self.p.alpha) for worker in self.workers]
            # if step % self.p.report == 0:
            elapsed = time.time() - start
            self.print_rmse(elapsed, step)
        # print('FINAL')
        # self.print_rmse(step)
        print('runtime:', time.time() - start)
        self.shutdown()


class AsyncManager(Manager):
    def run(self):
        [worker.run.remote() for worker in self.workers]
        start = time.time()
        for i in range(self.num_col_parts):
            self.pending.put(self.complete.get())
        while True:
            while not self.results.empty():
                nnz, total = self.results.get()
                # https://stats.stackexchange.com/a/221831
                # Double check this
                self.nnz += nnz
                self.rmse = ((self.nnz - nnz) / self.nnz) * self.rmse + total / self.nnz
            time.sleep(self.p.report)
            elapsed = time.time() - start
            if elapsed > self.p.d:
                break
            self.print_rmse(elapsed)
        # print('FINAL')
        # self.print_rmse()
        print('runtime:', time.time() - start)
        self.shutdown()
