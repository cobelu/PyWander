import multiprocessing
from typing import List

import numpy as np
import ray
import time
from ray.util.queue import Queue
from scipy.sparse import csr_matrix, load_npz

from parameters import Parameters
from partition import Partition
from util import load, load_with_features
from work import Work
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

        # Initialize the workers
        self.workers = [
            Worker.remote(
                i,
                p,
                self.pending,
                self.complete,
                self.results,
                a_csr[row_ptns[i].low:row_ptns[i].high].tocsc(),
                row_ptns[i],
                # col_ptns[i]
            ) for i in range(self.p.n)
        ]

        [self.pending.put(
            Work(
                ptn,
                1 / np.sqrt(self.p.k) * Worker.random((self.p.k, ptn.dim())),
                -1,
                0,
            )) for ptn in col_ptns]

        self.print_params()

    def run(self):
        raise NotImplementedError()

    def print_rmse(self, dur, step=None):
        rmse = np.sqrt(self.rmse / self.nnz) if self.nnz > 0 else np.NaN
        # print("\tNNZ: {0}".format(self.nnz))
        # print("\tRMSE: {0}".format(rmse))
        if step:
            print("{0}\t{1}\t{2}\t{3}".format(dur, self.nnz, rmse, step))
        else:
            print("{0}\t{1}\t{2}".format(dur, self.nnz, rmse))

    def print_params(self):
        print("sync\tn\td\tk\talpha\tbeta\tlamda\tptns\treport\tnormalize\tbold\tmethod\tdata")
        print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}".format(
            self.p.sync,
            self.p.n,
            self.p.d,
            self.p.k,
            self.p.alpha,
            self.p.beta,
            self.p.lamda,
            self.p.ptns,
            self.p.report,
            self.p.normalize,
            self.p.bold,
            self.p.method,
            self.p.filename),
        )
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
            # for i in range(self.num_col_parts):
            #     self.pending.put(self.complete.get())
            for i in range(self.p.d):
                nnz, total = self.results.get()
                # https://stats.stackexchange.com/questions/221826/is-it-possible-to-compute-rmse-iteratively
                # Double check this
                self.nnz += nnz
                self.rmse += total
                # self.rmse = ((self.nnz - nnz) / self.nnz) * self.rmse + total / self.nnz
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
        # Initial work is already placed on queue
        # for i in range(self.num_col_parts):
        #     self.pending.put(self.complete.get())
        start = time.time()
        shout = self.p.report
        # Start working
        while True:
            if not self.results.empty():
                nnz, total = self.results.get()
                # print("[NNZ: {0}, TOTAL: {1}".format(nnz, total))
                # https://stats.stackexchange.com/a/221831
                # Double check this
                self.nnz += nnz
                self.rmse += total
            elapsed = time.time() - start
            if elapsed >= self.p.d:
                self.print_rmse(elapsed)
                break
            if elapsed >= shout:
                self.print_rmse(elapsed)
                shout += self.p.report
        self.shutdown()
        return
