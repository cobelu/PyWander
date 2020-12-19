# Connor Luckett
# Wander
import argparse

import ray

from manager import SyncManager, AsyncManager
from parameters import Parameters


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', default=0.08, type=float)
    parser.add_argument('-b', '--beta', default=0.05, type=float)
    parser.add_argument('-l', '--lamda', default=1.0, type=float)
    parser.add_argument('-d', '--duration', default=10, type=int)
    parser.add_argument('-k', '--latent', default=100, type=int)
    parser.add_argument('-p', '--partitions', default=1, type=int)
    parser.add_argument('-r', '--report', default=1, type=int)
    parser.add_argument('-w', '--workers', default=1, type=int)
    parser.add_argument('-s', '--sync', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('filename')
    args = parser.parse_args()

    # Build parameters from the args
    k = args.latent
    alpha = args.alpha
    beta = args.beta
    lamda = args.lamda
    ptns = args.partitions
    report = args.report
    sync = args.sync
    duration = args.duration
    workers = args.workers
    filename = args.filename
    verbose = args.verbose
    p = Parameters(sync, workers, duration, k, alpha, beta, lamda, ptns, report, filename, verbose)

    # Go do stuff with Ray
    ray.init()
    if args.sync:
        SyncManager(p).run()
    else:
        AsyncManager(p).run()
    ray.shutdown()


if __name__ == '__main__':
    main()
