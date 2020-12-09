# Connor Luckett
# Wander
import sys
from random import shuffle

import ray
import signal
import numpy as np
from ray.util.queue import Queue

from nonsync import Async
from scheduler import Scheduler
from sync import Sync
from timeout import alarm_handler, TimeoutException
from work import Work


def main():
    k = 100
    alpha = 0.08
    beta = 0.05
    lamda = 1
    sync = True
    duration = 10
    workers = 4
    file = "data/netflix.npz"
    rows, cols, normalizer = Scheduler.load_dims(file)

    # print(sys.float_info)
    ray.init()
    first_work = Work(0, cols)
    works = first_work.splits(workers, True)
    print(', '.join(map(str, works)))
    hs = [None for _ in range(workers)]
    queues = [Queue() for _ in range(workers)]
    total = 0
    nnz = 0

    # Create the communicators
    if sync:
        schedulers = [Sync.remote(file, normalizer, works[i], queues) for i in range(workers)]
        readies = [scheduler.ready.remote() for scheduler in schedulers]
        print("Waiting...")
        ray.wait(readies, num_returns=workers)
        print("Ready!")
        for i in range(1, duration+1):
            print("Iteration: {0}".format(i))
            results = [schedulers[i].sgd.remote(works[i], hs[i]) for i in range(workers)]
            got_results = [ray.get(results[i], timeout=1000) for i in range(workers)]
            machines_total = sum([row[0] for row in got_results])
            machines_nnz = sum([row[1] for row in got_results])
            hs = [row[2] for row in got_results]
            total += machines_total
            nnz += machines_nnz
            print("RMSE: {0}".format(np.sqrt(total/nnz)))
    else:
        while True:
            schedulers = [Async.remote(file, w) for w in works]
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(duration)
            try:
                works = [schedulers[i].sgd.remote(works[i], None) for i in range(len(schedulers))]
            except TimeoutException:
                print("Timeout")
            finally:
                # Reset alarm clock
                signal.alarm(0)

    # Train
    # nxt = 0
    # count = 0
    # while count < duration:
    #     w = queue.get()
    #     schedulers[nxt].calc().remote(w)
    #     nxt = (nxt + 1) % workers

    ray.shutdown()


if __name__ == '__main__':
    main()
