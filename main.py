# Connor Luckett
# Wander
import sys
from random import shuffle

import ray
import signal

from nonsync import Async
from scheduler import Scheduler
from sync import Sync
from timeout import alarm_handler, TimeoutException
from work import Work


def main():
    # print(sys.float_info)
    ray.init()

    sync = True
    duration = 5
    workers = 4
    file = "data/netflix.npz"
    rows, cols, normalizer = Scheduler.load_dims(file)
    first_work = Work(0, cols)
    works = first_work.splits(workers, True)
    print(', '.join(map(str, works)))
    h = [None for _ in range(workers)]

    # Create the communicators
    if sync:
        schedulers = [Sync.remote(file, normalizer, w) for w in works]
        for i in range(1, duration+1):
            print("Iteration: {0}".format(i))
            results = [schedulers[i].sgd.remote(works[i], h[i]) for i in range(workers)]
            got_results = [ray.get(results[i], timeout=1000) for i in range(workers)]
            print("Results: {0})".format(got_results))
            # print("Results: (totals: {0}, nnzs: {1})".format(totals, nnzs))
    else:
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

    print(schedulers)

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
