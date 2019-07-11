""""Baseline Hidden Markov Model (HMM)."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import joblib
import pandas as pd
import numpy as np

from pomegranate import *
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

from datasets import seq_collate_dict, load_dataset

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def load_data(modalities, data_dir, normalize=[]):
    print("Loading data...")
    train_data = load_dataset(modalities, data_dir, 'Train',
                              base_rate=args.base_rate,
                              truncate=True, item_as_dict=True)
    test_data = load_dataset(modalities, data_dir, 'Valid',
                             base_rate=args.base_rate,
                             truncate=True, item_as_dict=True)
    print("Done.")
    if len(normalize) > 0:
        print("Normalizing ", normalize, "...")
        # Normalize test data using training data as reference
        test_data.normalize_(modalities=normalize, ref_data=train_data)
        # Normailze training data in-place
        train_data.normalize_(modalities=normalize)
    return train_data, test_data

def train(train_data, test_data, args):
    # Concatenate across input modalities for each training sequence
    X_train = [np.concatenate([seq[m] for m in args.modalities], axis=1)
               for seq in train_data]
    # Extract ratings as target outputs to fit against
    y_train = [seq['ratings'] for seq in train_data]
    # Concatenate inputs with targets for each sequence
    Xy_train = [np.concatenate([X_seq, y_seq], axis=1)
                for X_seq, y_seq in zip(X_train, y_train)]
    # Zero-mask missing values
    # X_train[np.isnan(X_train)] = 0.0
    # y_train[np.isnan(y_train)] = 0.0

    # Set up hyper-parameters for support vector regression
    params = {
        'n_components': [10]
    }
    params = list(ParameterGrid(params))

    # Cross validate across hyper-parameters
    best_ccc = -1
    for p in params:
        print("Using parameters:", p)

        # Learn HMM from training set
        print("Learning HMM from training data...")
        model = HiddenMarkovModel.\
                from_samples(MultivariateGaussianDistribution,
                             X=Xy_train, verbose=True, **p)

        # Evaluate on test set
        ccc, predictions = evaluate(model, test_data, args)

        # Save best parameters and model
        if ccc > best_ccc:
            best_ccc = ccc
            best_params = p 
            best_model = model
            best_pred = predictions

    # Print best parameters
    print('---')
    print('Best CCC: {:0.3f}'.format(best_ccc))
    print('Best parameters:', best_params)

    return best_ccc, best_params, best_model, best_pred

def evaluate(model, test_data, args, fig_path=None):
    ccc = []
    predictions = []
    
    # Predict and evaluate on each test sequence
    print("Evaluating...")
    for i, seq in enumerate(test_data):
        # Contatenate input modalities
        X_test = np.concatenate([seq[m] for m in args.modalities], axis=1)
        # Concatenate inputs with NaN targets
        y_nan = np.ones((len(X_test), 1)) * np.nan
        Xy_test = np.concatenate([X_test, y_nan], axis=1)
        # Get ground truth ratings
        y_test = test_data.orig['ratings'][i].flatten()

        # Infer most likely hidden states
        _, state_path = model.viterbi(Xy_test)
        print(len(state_path), len(X_test))
        # Predict most likely ratings (Gaussian mean) from states
        y_pred = [state.distribution.parameters[0][-1]
                  for i, state in state_path[1:]]

        # Ensure predictions and ground truth are at same sampling rate
        if test_data.ratios['ratings'] >= 1:
            # Repeat and pad predictions to match original data length
            ratio = test_data.ratios['ratings']
            y_pred = np.repeat(y_pred, ratio)[:len(y_test)]
            l_diff = len(y_test) - len(y_pred)
            if l_diff > 0:
                y_pred = np.concatenate([y_pred, y_pred[-l_diff:]])
        elif test_data.ratios['ratings'] < 1:
            # Time average to match original data length
            ratio = int(np.round(1 / test_data.ratios['ratings']))
            end = min(ratio * len(y_test),
                      ratio * (len(y_pred) // ratio))
            avg = np.mean(y_pred[:end].reshape(-1, ratio, y_pred.shape[1]), 1)
            if end < (ratio * len(y_test)):
                remain = y_pred[end:].mean(axis=0)[np.newaxis,:]
                y_pred = np.concatenate([avg, remain])
            else:
                y_pred = avg

        # Smoothing predictions via moving average
        # y_pred = moving_average(y_pred, args.sma)
                
        ccc.append(eval_ccc(y_test, y_pred))
        predictions.append(y_pred)
        
    # Visualize predictions
    if args.visualize:
        plot_predictions(test_data, predictions, ccc, args, fig_path)
        
    ccc_std = np.std(ccc)
    ccc = np.mean(ccc)
    print('CCC: {:0.3f} +-{:0.3f}'.format(ccc, ccc_std))
    return ccc, predictions

def plot_predictions(dataset, predictions, metric, args, fig_path=None):
    """Plots predictions against ratings for representative fits."""
    # Create figure to visualize predictions
    if not hasattr(args, 'fig'):
        args.fig, args.axes = plt.subplots(4, 2, figsize=(6,8))

    # Select top 4 and bottom 4
    sel_idx = np.concatenate((np.argsort(metric)[-4:][::-1],
                              np.argsort(metric)[:4]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_true = [dataset.orig['ratings'][i] for i in sel_idx]
    sel_pred = [predictions[i] for i in sel_idx]
    for i, (true, pred, m) in enumerate(zip(sel_true, sel_pred, sel_metric)):
        j, i = (i // 4), (i % 4)
        args.axes[i,j].cla()
        args.axes[i,j].plot(true, 'b-')
        args.axes[i,j].plot(pred, 'c-')
        args.axes[i,j].set_xlim(0, len(true))
        args.axes[i,j].set_ylim(-1, 1)
        args.axes[i,j].set_title("Fit = {:0.3f}".format(m))
    plt.tight_layout()
    plt.draw()
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.pause(1.0 if (args.load is not None) else 0.001)

def save_predictions(dataset, predictions, path):
    for p, seq_id in zip(predictions, dataset.seq_ids):
        df = pd.DataFrame(p, columns=['rating'])
        fname = "target_{}_{}_normal.csv".format(*seq_id)
        df.to_csv(os.path.join(path, fname), index=False)
        
def main(args):
    # Construct modality names if not provided
    if args.modalities is None:
        args.modalities = ['acoustic', 'linguistic', 'emotient']

    # Load data for specified modalities
    train_data, test_data = load_data(args.modalities, args.data_dir,
                                      args.normalize)
    print('---')

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.load is not None:
        # Load saved model
        with open(args.load, 'r') as model_file:
            model = HiddenMarkovModel.from_json(model_file.read())
    elif args.test:
        # Load best model in save directory
        with open(os.path.join(args.save_dir, "best.sav"), 'r') as model_file:
            model = HiddenMarkovModel.from_json(model_file.read())
    else:
        # Fit new model against training data, cross-val against test data
        _, _, model, _ = train(train_data, test_data, args)
        model_json = model.to_json()
        with open(os.path.join(args.save_dir, "best.sav"), 'w+') as model_file:
            model_file.write(model_json)

    # Create paths to save predictions
    pred_train_dir = os.path.join(args.save_dir, "pred_train")
    pred_test_dir = os.path.join(args.save_dir, "pred_test")
    if not os.path.exists(pred_train_dir):
        os.makedirs(pred_train_dir)
    if not os.path.exists(pred_test_dir):
        os.makedirs(pred_test_dir)

    # Evaluate model on training and test set
    print("-Training-")
    ccc1, pred = evaluate(model, train_data, args,
                          os.path.join(args.save_dir, "train.png"))
    save_predictions(train_data, pred, pred_train_dir)
    print("-Testing-")
    ccc2, pred = evaluate(model, test_data, args,
                          os.path.join(args.save_dir, "test.png"))
    save_predictions(test_data, pred, pred_test_dir)
    return ccc1, ccc2
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modalities', type=str, default=None, nargs='+',
                        help='input modalities (default: all)')
    parser.add_argument('--normalize', type=str, default=[], nargs='+',
                        help='modalities to normalize (default: [])')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='flag to visualize predictions (default: false)')
    parser.add_argument('--sma', type=int, default=1, metavar='W',
                        help='window size for moving average (default: 1)')
    parser.add_argument('--load', type=str, default=None,
                        help='path to trained model to evaluate')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training (default: false)')
    parser.add_argument('--base_rate', type=float, default=2.0, metavar='N',
                        help='sampling rate to resample to (default: 2.0)')
    parser.add_argument('--data_dir', type=str, default="../../data",
                        help='path to data base directory')
    parser.add_argument('--save_dir', type=str, default="./hmm_save",
                        help='path to save models and predictions')
    args = parser.parse_args()
    main(args)
