"""Script to run Ray Tune experiments in parallel."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, argparse
from itertools import chain, combinations

import ray
import ray.tune as tune

from train import tuner

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_repeats', type=int, default=1, metavar='N',
                        help='number of repetitions per config set')
    parser.add_argument('--trial_cpus', type=int, default=1, metavar='N',
                        help='number of CPUs per trial')
    parser.add_argument('--trial_gpus', type=int, default=0, metavar='N',
                        help='number of GPUs per trial')
    parser.add_argument('--max_cpus', type=int, default=None, metavar='N',
                        help='max CPUs for all trials')
    parser.add_argument('--max_gpus', type=int, default=None, metavar='N',
                        help='max GPUs for all trials')
    parser.add_argument('--data_dir', type=str, default="../../data",
                        help='path to data base directory')
    args = parser.parse_args()

    # If max resources not specified, default to maximum - 1
    if args.max_cpus is None:
        import psutil
        args.max_cpus = min(1, psutil.cpu_count() - 1)
    if args.max_gpus is None:
        import torch
        args.max_gpus = min(1, torch.cuda.device_count() - 1)
    
    ray.init(num_cpus=args.max_cpus, num_gpus=args.max_gpus)

    # Convert data dir to absolute path so that Ray trials can find it
    data_dir = os.path.abspath(args.data_dir)

    # Generate all possible combinations of modalities
    mod_combs = powerset(['acoustic', 'linguistic', 'emotient'])
    mod_combs = [list(mods) for mods in mod_combs if len(mods) > 0]
    
    trials = tune.run(
        tuner,
        name="sync_tune",
        config={
            "data_dir": data_dir,
            # Repeat each configuration with different random seeds
            "seed": tune.grid_search(range(args.n_repeats)),
            # Iterate over all possible combinations of modalities
            "modalities": tune.grid_search(mod_combs)
        },
        local_dir="./",
        resources_per_trial={"cpu": args.trial_cpus, "gpu": args.trial_gpus}
    )
