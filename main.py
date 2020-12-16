# Connor Luckett
# Wander
import sys
from random import shuffle

import ray
import signal
import numpy as np

from ray.util.queue import Queue

from parameters import Parameters
from scheduler import Scheduler
from timeout import alarm_handler, TimeoutException
from work import Work


def main():
    k = 100
    alpha = 0.08
    beta = 0.05
    lamda = 1
    sync = True
    duration = 1
    workers = 4
    file = "data/netflix.npz"
    rows, cols, normalizer = Scheduler.load_dims(file)
    p = Parameters(sync, workers, duration, k, alpha, beta, lamda, normalizer, file)

    ray.init()
    row_works = Work(0, rows).splits(workers, True)
    col_works = Work(0, cols).splits(workers, True)
    hs = [None for _ in range(workers)]
    queues = [Queue() for _ in range(workers)]
    schedulers = [Scheduler.remote(i, p, row_works[i], queues) for i in range(workers)]

    dumpeds = [schedulers[i].dump.remote([col_works[i]]) for i in range(workers)]
    ray.wait(dumpeds, num_returns=workers)

    total = 0
    nnz = 0
    if sync:
        readies = [scheduler.ready.remote() for scheduler in schedulers]
        print("Waiting...")
        ray.wait(readies, num_returns=workers)
        print("Ready!")
        for step in range(1, duration+1):
            print("Iteration: {0}".format(step))
            results = [schedulers[i].sgd.remote(hs[i]) for i in range(workers)]
            got_results = [ray.get(results[i], timeout=10000) for i in range(workers)]
            worker_totals = sum([row[0] for row in got_results])
            worker_nnzs = sum([row[1] for row in got_results])
            hs = [row[2] for row in got_results]
            total += worker_totals
            nnz += worker_nnzs
            print("NNZ: {0}".format(nnz))
            print("RMSE: {0}".format(np.sqrt(total/nnz)))
    else:
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

    ray.shutdown()


if __name__ == '__main__':
    main()
