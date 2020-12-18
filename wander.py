# Connor Luckett
# Wander
import argparse
import ray
import signal
import sys

from manager import AsyncManager, SyncManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', default=0.08, type=float)
    parser.add_argument('-b', '--beta', default=0.05, type=float)
    parser.add_argument('-d', '--duration', default=10, type=int)
    parser.add_argument('-k', '--latent', default=100, type=int)
    parser.add_argument('-l', '--lamda', default=1.0, type=float)
    parser.add_argument('-p', '--partitions', default=1, type=int)
    parser.add_argument('-r', '--report', default=1, type=int)
    parser.add_argument('-s', '--sync', action='store_true')
    parser.add_argument('-w', '--workers', default=1, type=int)
    parser.add_argument('filename')
    args = parser.parse_args()

    ray.init()
    if args.sync:
        SyncManager(args).run()
    else:
        AsyncManager(args).run()
    ray.shutdown()

if __name__ == '__main__':
    main()
