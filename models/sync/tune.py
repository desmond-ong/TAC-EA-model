"""Script to run Ray Tune experiments in parallel."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, argparse
from itertools import chain, combinations

import pandas as pd
import ray
import ray.tune as tune

from analysis import ExperimentAnalysis
import train

parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--analyze', action='store_true', default=False,
                    help='analyze without training (default: false)')
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
parser.add_argument('--local_dir', type=str, default="./",
                    help='path to Ray results')
parser.add_argument('--exp_name', type=str, default="sync_tune",
                    help='experiment name')

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def run(args):
    """Runs Ray experiments."""
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
        train.tuner,
        name=args.exp_name,
        config={
            "lr": 5e-6,
            "data_dir": data_dir,
            # Repeat each configuration with different random seeds
            "seed": tune.grid_search(range(args.n_repeats)),
            # Iterate over all possible combinations of modalities
            "modalities": tune.grid_search(mod_combs)
        },
        local_dir=args.local_dir,
        resources_per_trial={"cpu": args.trial_cpus, "gpu": args.trial_gpus}
    )

def analyze(args):
    """Analyze and run saved models on test set."""
    exp_dir = os.path.join(args.local_dir, args.exp_name)
    ea = ExperimentAnalysis(exp_dir)
    df = ea.dataframe()
    best_train_stats = []
    best_test_stats = []
    
    # Iterate across trials
    for trial in ea._checkpoints:
        print("Trial:", trial['experiment_tag'])
        config = trial['config']
        trial_dir = os.path.basename(trial['logdir'])
        trial_dir = os.path.join(exp_dir, trial_dir)
        
        # Set up parameter namespace with default arguments
        eval_args = train.parser.parse_args([])
        # Override args with config
        vars(eval_args).update(config)
        # Set save directory
        eval_args.save_dir = os.path.join(trial_dir, eval_args.save_dir)
        # Set to test
        eval_args.test = True

        # Evaluate on train and test set
        train_stats, test_stats = train.main(eval_args)
        train_stats = {'best_train_'+k : v for k, v in train_stats.iteritems()}
        test_stats = {'best_test_'+k : v for k, v in test_stats.iteritems()}
        best_train_stats.append(train_stats)
        best_test_stats.append(test_stats)

    # Add to dataframe and save
    train_df = pd.DataFrame(best_train_stats)
    test_df = pd.DataFrame(best_test_stats)
    df = pd.concat([df, train_df, test_df], axis=1)
    df = df.sort_values(['experiment_tag'])
    df.to_csv(os.path.join(exp_dir, 'analysis.csv'), index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.analyze:
        run(args)
    analyze(args)
