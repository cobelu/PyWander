# Connor Luckett
# Wander
import ray
import signal
import numpy as np

from nonsync import Async
from scheduler import Scheduler
from sync import Sync
from timeout import alarm_handler, TimeoutException
from work import Work
from ray.util.queue import Queue


def main():
    ray.init()

    sync = True
    duration = 1
    workers = 4
    file = "data/netflix.npz"
    work = Work(0, 17000)
    works = work.splits(workers)
    queue = Queue()
    for w in works:
        queue.put(w)

    # Create the communicators
    if sync:
        schedulers = [Sync.remote(file, w) for w in works]
        for i in range(duration):
            print("Iteration:", i)
    else:
        schedulers = [Async.remote(file, w) for w in works]
        # signal.signal(signal.SIGALRM, alarm_handler)
        # signal.alarm(duration)
        # try:
        #     print("Running code")
        # except TimeoutException:
        #     print("Timeout")
        # finally:
        #     # Reset alarm clock
        #     signal.alarm(0)

    print(schedulers)

    # Train
    # nxt = 0
    # count = 0
    # while count < duration:
    #     w = queue.get()
    #     schedulers[nxt].calc().remote(w)
    #     nxt = (nxt + 1) % workers

    works = [scheduler.sgd.remote(Work(0, 1), np.array(0)) for scheduler in schedulers]
    results = ray.get(works)
    print(results)


if __name__ == '__main__':
    main()
