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

    sync = False
    duration = 1
    workers = 4
    file = "data/netflix.npz"
    shape = Scheduler.load_dims(file)
    num_rows, num_cols = shape[0], shape[1]
    work = Work(0, num_rows)
    works = work.splits(workers)
    print("Works:")
    for w in works:
        print(w)
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

    works = [scheduler.sgd.remote(Work(0, 1), None) for scheduler in schedulers]
    results = ray.get(works, timeout=1000)
    print("Results:\n{0}".format(results))

    ray.shutdown()


if __name__ == '__main__':
    main()
