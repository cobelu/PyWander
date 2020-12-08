# Connor Luckett
# Wander
from random import shuffle

import ray
import signal

from nonsync import Async
from scheduler import Scheduler
from sync import Sync
from timeout import alarm_handler, TimeoutException
from work import Work


def main():
    ray.init()

    sync = True
    duration = 3
    workers = 4
    file = "data/netflix.npz"
    rows, cols = Scheduler.load_dims(file)
    first_work = Work(0, cols)
    works = first_work.splits(workers, True)

    # Create the communicators
    if sync:
        schedulers = [Sync.remote(file, w) for w in works]
        for i in range(duration):
            works = [schedulers[i].sgd.remote(works[i], None) for i in range(len(schedulers))]
            shuffle_wo_repeat(works)
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
    results = ray.get(works, timeout=1000)
    print("Results:\n{0}".format(results))

    ray.shutdown()


def shuffle_wo_repeat(some_list):
    """
    https://stackoverflow.com/a/15512349

    :param some_list: A list to be shuffled
    :return: The shuffled list without repeat
    """
    randomized_list = some_list[:]
    while True:
        shuffle(randomized_list)
        for a, b in zip(some_list, randomized_list):
            if a == b:
                break


if __name__ == '__main__':
    main()
