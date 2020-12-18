from typing import List

import numpy as np
from ray.util.queue import Queue

from parameters import Parameters
from partition import Partition
from worker import Worker


class Manager:
    def __init__(self, p: Parameters):
        self.p = p
        self.total = 0
        self.nnz = 0
        self.pending = Queue()
        self.complete = Queue()
        row_ptns = self.handouts()
        self.workers = [Worker.remote(i, p, self.pending, self.complete, row_ptns[i])
                        for i in range(self.p.n)]

    def run(self):
        raise NotImplementedError()

    def handouts(self) -> List[Partition]:
        return Partition(0, self.p.rows).splits(self.p.n, False)

    def print_rmse(self):
        print("NNZ: {0}".format(self.nnz))
        print("RMSE: {0}".format(np.sqrt(self.total / self.nnz)))


class SyncManager(Manager):
    def run(self):
        # Run every iteration
        for step in range(1, self.p.d + 1):
            # Place initial work on queue
            for i in range(len(self.workers)):
                self.pending.put(self.complete.get())
            # Put sentinel work on queue
            for i in range(len(self.workers)):
                self.pending.put(None)
            print("Iteration: {0}".format(step))
            # Start calculating
            [worker.run.remote() for worker in self.workers]
            while not self.pending.empty():
                pass
            for worker in self.workers:
                total, nnz = worker.update()
                self.total += total
                self.nnz += nnz
            if step % self.p.report == 0:
                self.print_rmse()
        # Print final RMSE
        self.print_rmse()


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
