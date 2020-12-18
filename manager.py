import numpy as np
import random
import ray
from ray.util.queue import Queue

from partition import Partition
from worker import Worker

class Manager:
    def __init__(self, args):
        self.args = args
        self.total = 0
        self.nnz = 0
        self.pending = Queue()
        self.complete = Queue()
        self.workers = [Worker.remote(i, args, self.pending, self.complete)
                        for i in range(self.args.workers)]

    def run(self):
        raise NotImplementedError()

    def print_rmse(self):
        print("NNZ: {0}".format(self.nnz))
        print("RMSE: {0}".format(np.sqrt(self.total / self.nnz)))

class SyncManager(Manager):
    def run(self):
        for iter in range(1, self.args.duration + 1):
            for i in range(len(self.workers)):
                self.pending.put(self.complete.get())
            for i in range(len(self.workers)):
                self.pending.put(None)

            print("Iteration: {0}".format(iter))
            for worker in self.workers:
                worker.run.remote()

            while not self.pending.empty():
                pass

            for worker in self.workers:
                total, nnz = worker.update()
                self.total += total
                self.nnz += nnz

            if iter % self.args.report == 0:
                print_rmse()

        print_rmse()

class AsyncManager(Manager):
    def run(self):
        for i in range(len(self.workers)):
            self.pending.put(self.complete.get())
        pass
        """
        for h in self.hs:
            self.pending.put(h)
        while True:
            schedulers = [Scheduler.remote(file, w) for w in row_works]
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(duration)
            try:
                row_works = [schedulers[i].sgd.remote(row_works[i], None) for i in range(len(schedulers))]
            except TimeoutException:
                print("Timeout")
            finally:
                # Reset alarm clock
                signal.alarm(0)
        """
