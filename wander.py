# Connor Luckett
# Wander
from nonsync import Async
from sync import Sync


def main():
    sync = True
    i = 1

    # Create the communicators
    if sync:
        scheduler = Sync()
    else:
        scheduler = Async()

    # Train
    for iteration in range(i):
        scheduler.sgd()


if __name__ == '__main__':
    main()
