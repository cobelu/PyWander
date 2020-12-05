# Connor Luckett
# Wander
import ray

from nonsync import Async
from scheduler import Scheduler
from sync import Sync
from work import Work
from ray.util.queue import Queue


def main():
    ray.init()

    sync = True
    duration = 1
    workers = 4
    file = "data/netflix.npz"
    work = Work(0, 1)
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

    print(schedulers)

    # Train
    # nxt = 0
    # count = 0
    # while count < duration:
    #     w = queue.get()
    #     schedulers[nxt].calc().remote(w)
    #     nxt = (nxt + 1) % workers


if __name__ == '__main__':
    main()
