"""Training code for synchronous multimodal LSTM model."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import seq_collate_dict, load_dataset
from models import MultiLSTM, MultiEDLSTM, MultiARLSTM

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def train(loader, model, criterion, optimizer, epoch, args):
    model.train()
    data_num = 0
    loss= 0.0
    log_freq = len(loader) // args.log_freq
    # Select batches that should be supervised
    supervised = np.zeros(len(loader), dtype=bool)
    supervised[:int(args.sup_ratio * len(loader))].fill(True)
    np.random.shuffle(supervised)
    for batch_num, (data_dict, mask, lengths) in enumerate(loader):
        # Send to device
        mask = mask.to(args.device)
        for m in data_dict.keys():
            data_dict[m] = data_dict[m].to(args.device)
        # Set whether to use teacher forcing
        target = data_dict['ratings'] if supervised[batch_num] else None
        # Run forward pass.
        output = model(data_dict, mask, lengths, target=target)
        # Compute loss and gradients
        batch_loss = criterion(output, data_dict['ratings'])
        # Accumulate total loss for epoch
        loss += batch_loss
        # Average over number of non-padding datapoints before stepping
        batch_loss /= sum(lengths)
        batch_loss.backward()
        # Step, then zero gradients
        optimizer.step()
        optimizer.zero_grad()
        # Keep track of total number of time-points
        data_num += sum(lengths)
        if batch_num % log_freq == 0:
            print('Batch: {:5d}\tLoss: {:2.5f}'.\
                  format(batch_num, loss/data_num))
    # Average losses and print
    loss /= data_num
    print('---')
    print('Epoch: {}\tLoss: {:2.5f}'.format(epoch, loss))
    return loss

def evaluate(dataset, model, criterion, args, fig_path=None):
    model.eval()
    predictions = []
    data_num = 0
    loss, corr, ccc = 0.0, [], []
    for data, orig in zip(dataset, dataset.orig['ratings']):
        # Collate data into batch dictionary of size 1
        data_dict, mask, lengths = seq_collate_dict([data])
        # Send to device
        mask = mask.to(args.device)
        for m in data_dict.keys():
            data_dict[m] = data_dict[m].to(args.device)
        # Run forward pass
        output = model(data_dict, mask, lengths)
        # Compute loss
        loss += criterion(output, data_dict['ratings'])
        # Keep track of total number of time-points
        data_num += data['length']
        # Resize predictions to match original length
        pred = output[0,:data['length']].view(-1).cpu().numpy()
        pred = np.repeat(pred, int(dataset.ratios['ratings']))[:len(orig)]
        if len(pred) < len(orig):
            pred = np.concatenate((pred, pred[len(pred)-len(orig):]))
        predictions.append(pred)
        # Compute correlation and CCC of predictions against ratings
        corr.append(pearsonr(orig.reshape(-1), pred)[0])
        ccc.append(eval_ccc(orig.reshape(-1), pred))
    # Plot predictions against ratings
    if args.visualize:
        plot_predictions(dataset, predictions, ccc, args, fig_path)
    # Average losses and print
    loss /= data_num
    corr = sum(corr) / len(corr)
    ccc = sum(ccc) / len(ccc)
    print('Evaluation\tLoss: {:2.5f}\tCorr: {:0.3f}\tCCC: {:0.3f}'.\
          format(loss, corr, ccc))
    return predictions, loss, corr, ccc

def plot_predictions(dataset, predictions, metric, args, fig_path=None):
    """Plots predictions against ratings for representative fits."""
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
    plt.pause(1.0 if args.test else 0.001)

def save_predictions(dataset, predictions, path):
    for p, seq_id in zip(predictions, dataset.seq_ids):
        df = pd.DataFrame(p, columns=['rating'])
        fname = "target_{}_{}_normal.csv".format(*seq_id)
        df.to_csv(os.path.join(path, fname), index=False)

def save_params(args, model, train_ccc, test_ccc):
    fname = 'param_hist.tsv'
    df = pd.DataFrame([vars(args)], columns=vars(args).keys())
    df = df[['modalities', 'batch_size', 'split', 'epochs', 'lr',
             'sup_ratio', 'base_rate']]
    df.insert(0, 'test_ccc', [test_ccc])
    df.insert(0, 'train_ccc', [train_ccc])
    df.insert(0, 'model', [model.__class__.__name__])
    df['embed_dim'] = model.embed_dim
    df['h_dim'] = model.h_dim
    df['attn_len'] = model.attn_len
    if type(model) is MultiARLSTM:
        df['ar_order'] = [model.decay.item()]
    else:
        df['ar_order'] = [float('nan')]
    df.set_index('model')
    df.to_csv(fname, mode='a', header=(not os.path.exists(fname)), sep='\t')
        
def save_checkpoint(modalities, model, path):
    checkpoint = {'modalities': modalities, 'model': model.state_dict()}
    torch.save(checkpoint, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

def load_data(modalities, data_dir):
    print("Loading data...")
    train_data = load_dataset(modalities, data_dir, 'Train',
                              base_rate=args.base_rate,
                              truncate=True, item_as_dict=True)
    test_data = load_dataset(modalities, data_dir, 'Valid',
                             base_rate=args.base_rate,
                             truncate=True, item_as_dict=True)
    print("Done.")
    return train_data, test_data

def main(args):
    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    # Convert device string to torch.device
    args.device = (torch.device(args.device) if torch.cuda.is_available()
                   else torch.device('cpu'))

    # Load model if specified, or test/feature flags are set
    checkpoint = None
    if args.load is not None:
        checkpoint = load_checkpoint(args.load, args.device)
    elif args.test:
        # Load best model in output directory if unspecified
        model_path = os.path.join(args.save_dir, "best.pth")
        checkpoint = load_checkpoint(model_path, args.device)
    
    if checkpoint is not None:
        # Use loaded modalities
        args.modalities = checkpoint['modalities']
    elif args.modalities is None:
        # Default to acoustic if unspecified
        args.modalities = ['acoustic', 'linguistic', 'emotient']

    # Load data for specified modalities
    train_data, test_data = load_data(args.modalities, args.data_dir)
    
    # Construct multimodal LSTM model
    dims = {'acoustic': 988, 'linguistic': 300, 'emotient': 20}
    model = MultiEDLSTM(args.modalities,
                        dims=(dims[m] for m in args.modalities),
                        device=args.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    # Setup loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Create path to save models/predictions
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create figure to visualize predictions
    if args.visualize:
        args.fig, args.axes = plt.subplots(4, 2, figsize=(6,8))
        
    # Evaluate model if test flag is set
    if args.test:
        # Create paths to save predictions
        pred_train_dir = os.path.join(args.save_dir, "pred_train")
        pred_test_dir = os.path.join(args.save_dir, "pred_test")
        if not os.path.exists(pred_train_dir):
            os.makedirs(pred_train_dir)
        if not os.path.exists(pred_test_dir):
            os.makedirs(pred_test_dir)
        # Evaluate on both training and test set
        with torch.no_grad():
            pred, _, _, ccc1 = evaluate(train_data, model, criterion, args,
                os.path.join(args.save_dir, "train.png"))
            save_predictions(train_data, pred, pred_train_dir)
            pred, _, _, ccc2 = evaluate(test_data, model, criterion, args,
                os.path.join(args.save_dir, "test.png"))
            save_predictions(test_data, pred, pred_test_dir)
        # Save command line flags, model params and CCC value
        save_params(args, model, ccc1, ccc2)
        return ccc1, ccc2

    # Split training data into chunks
    train_data = train_data.split(args.split)
    # Batch data using data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=seq_collate_dict,
                              pin_memory=True)
   
    # Train and save best model
    best_ccc = -1
    for epoch in range(1, args.epochs + 1):
        print('---')
        train(train_loader, model, criterion, optimizer, epoch, args)
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                pred, loss, corr, ccc =\
                    evaluate(test_data, model, criterion, args)
            if ccc > best_ccc:
                best_ccc = ccc
                path = os.path.join(args.save_dir, "best.pth") 
                save_checkpoint(args.modalities, model, path)
        # Save checkpoints
        if epoch % args.save_freq == 0:
            path = os.path.join(args.save_dir,
                                "epoch_{}.pth".format(epoch)) 
            save_checkpoint(args.modalities, model, path)

    # Save final model
    path = os.path.join(args.save_dir, "last.pth") 
    save_checkpoint(args.modalities, model, path)

    # Save command line flags, model params and CCC value
    save_params(args, model, float('nan'), best_ccc)
    
    return best_ccc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modalities', type=str, default=None, nargs='+',
                        help='input modalities (default: all')
    parser.add_argument('--batch_size', type=int, default=25, metavar='N',
                        help='input batch size for training (default: 25)')
    parser.add_argument('--split', type=int, default=1, metavar='N',
                        help='sections to split each video into (default: 1)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--sup_ratio', type=float, default=0.5, metavar='F',
                        help='teacher-forcing ratio (default: 0.5)')
    parser.add_argument('--base_rate', type=float, default=2.0, metavar='N',
                        help='sampling rate to resample to (default: 2.0)')
    parser.add_argument('--log_freq', type=int, default=5, metavar='N',
                        help='print loss N times every epoch (default: 5)')
    parser.add_argument('--eval_freq', type=int, default=1, metavar='N',
                        help='evaluate every N epochs (default: 1)')
    parser.add_argument('--save_freq', type=int, default=10, metavar='N',
                        help='save every N epochs (default: 10)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use (default: cuda:0 if available)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='flag to visualize predictions (default: false)')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs (default: false)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training (default: false)')
    parser.add_argument('--load', type=str, default=None,
                        help='path to trained model (either resume or test)')
    parser.add_argument('--data_dir', type=str, default="../../data",
                        help='path to data base directory')
    parser.add_argument('--save_dir', type=str, default="./lstm_save",
                        help='path to save models and predictions')
    args = parser.parse_args()
    main(args)
