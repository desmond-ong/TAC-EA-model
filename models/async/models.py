from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from scipy import integrate

def intensity_nll(t_diff, score, decay):
    """Computes negative log-likelihood of observing t_diff
       given the intensity function parameterized by score and decay."""
    log_intensity = score - decay * t_diff
    intensity = torch.exp(log_intensity)
    nll = -(log_intensity - 1/decay * (torch.exp(score) - intensity))
    return nll

def t_diff_mean(score, decay, t_max=None, t_step=None):
    """Computes expected value for t_diff given score and decay."""
    if t_max is None:
        t_max = 5.5 / decay.item()
    if t_step is None:
        t_step = 0.5 / decay.item()
    t_range = torch.arange(0, t_max, t_step, device=score.device)
    integrand = torch.cat([t * torch.exp(-intensity_nll(t, score, decay))
                           for t in t_range], dim=2)
    t_diff = np.trapz(integrand.detach().cpu().numpy(), dx=t_step, axis=2)
    return torch.tensor(t_diff).unsqueeze(-1).to(score.device)

def pad_shift(x, shift, padv=0.0):
    """Shift (batch, time, dims) tensor forwards in time with padding."""
    if shift > 0:
        padding = torch.ones(x.size(0), shift, x.size(2)).to(x.device) * padv
        return torch.cat((padding, x[:, :-shift, :]), dim=1)
    elif shift < 0:
        padding = torch.ones(x.size(0), -shift, x.size(2)).to(x.device) * padv
        return torch.cat((x[:, -shift:, :], padding), dim=1)
    else:
        return x
    
class SemiParamNPP(nn.Module):
    """Multimodal semi-parametric neural point process (NPP) model.

    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    h_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    """
    
    def __init__(self, modalities, dims, embed_dim=128, h_dim=512,
                 n_layers=1, decay=1.0, device=torch.device('cuda:0')):
        super(SemiParamNPP, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        
        # Create raw-to-embed FC+Dropout layer for each modality
        self.embed = dict()
        for m in self.modalities:
            self.embed[m] = nn.Sequential(nn.Dropout(0.1),
                                          nn.Linear(self.dims[m], embed_dim),
                                          nn.ReLU())
            self.add_module('embed_{}'.format(m), self.embed[m])
        # LSTM computes hidden states from embeddings and inter-event times
        self.lstm = nn.LSTM(1 + self.n_mods * embed_dim, h_dim,
                            n_layers, batch_first=True)
        # Network from LSTM hidden states to conditional intensity score
        self.time_out = nn.Sequential(nn.Linear(h_dim, embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(embed_dim, 1))
        # Regression network from LSTM hidden states to predicted value
        self.val_out = nn.Sequential(nn.Linear(h_dim, embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(embed_dim, 1))
        # Decay constant for intensity over time
        self.decay = nn.Parameter(torch.tensor(decay))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, mask, lengths, estimate=False):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Convert NaNs to zeros so they don't affect embedding value
        for m in self.modalities:
            inputs[m][torch.isnan(inputs[m])] = 0
        # Convert raw features into equal-dimensional embeddings
        embed = torch.cat([self.embed[m](inputs[m].view(-1, self.dims[m]))
                           for m in self.modalities], dim=1)
        # Compute inter-event times, and concatenate to embeddings
        t_diff = inputs['time'] - pad_shift(inputs['time'], 1)
        embed = torch.cat([t_diff.view(-1, 1), embed], dim=1)
        # Unflatten temporal dimension
        embed = embed.reshape(batch_size, seq_len, -1)
        # Pack the input to mask padded entries
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        # Forward propagate LSTM
        h, _ = self.lstm(embed)
        # Undo the packing and flatten temporal dimension
        h, _ = pad_packed_sequence(h, batch_first=True)
        h = h.reshape(-1, self.h_dim)
        # Decode the hidden state to time intensity scores annd event values
        score = self.time_out(h).view(batch_size, seq_len, 1)
        value = self.val_out(h).view(batch_size, seq_len, 1)
        # Mask entries that exceed sequence lengths
        score = score * mask.float()
        value = value * mask.float()
        if not estimate:
            # Return scores and predicted values if not estimating time
            return score, value
        else:
            # Otherwise predict expected value of timestamps using score
            return self.estimate(inputs['time'], score, value, mask)
            
    def estimate(self, time, score, value, mask):
        # Compute expected time differences
        t_diff = t_diff_mean(score, torch.abs(self.decay))
        # Sum to get next event times
        t_hat = time + t_diff
        # Iterate backwards and keep only prediced timestamps that are
        # smaller than the smallest timestamp seen so far
        t_min = t_hat[:,-1,:]
        keep = [mask[:,-1,:]]
        for i in reversed(range(0, t_hat.shape[1]-1)):
            k = mask[:,i,:] * (t_hat[:,i,:] < t_min)
            t_min = torch.min(t_min, t_hat[:,i,:])            
            keep.append(k)
        keep = torch.cat(keep, dim=1).unsqueeze(-1)
        # Filter out late event times and predicted values
        t_filt, val_filt = [], []
        for seq_id in range(keep.shape[0]):
            t_seq, val_seq = t_hat[seq_id,:,:], value[seq_id,:,:]
            t_filt.append(t_seq[keep[seq_id,:,:]])
            val_filt.append(val_seq[keep[seq_id,:,:]])
        return t_filt, val_filt
            
    def loss(self, data, score, value, mask, lambda_t=0.1):
        t_target, v_target = data['t_target'], data['v_target']
        time = data['time']
        # Find indices before last target is observed
        observed = (1 - torch.isnan(v_target)) * mask
        n_obs = observed.sum().item()
        if n_obs == 0:
            return torch.tensor(0.0).to(self.device), 0
        # Compute time differences from target observations
        t_diff = t_target - time
        # Compute intensity loss
        decay = torch.abs(self.decay)
        loss = (intensity_nll(t_diff[observed], score[observed], decay)).sum()
        loss *= lambda_t
        # Compute mean square loss between predicted values and targets
        loss += torch.sum((v_target[observed] - value[observed])**2)
        # Divide loss by number of non-padding datapoints
        loss /= n_obs
        return loss, n_obs

class NonParamNPP(nn.Module):
    """Multimodal non-parametric neural point process (NPP) model.

    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    h_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    """
    
    def __init__(self, modalities, dims, embed_dim=128, h_dim=512,
                 n_layers=1, device=torch.device('cuda:0')):
        super(NonParamNPP, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        
        # Create raw-to-embed FC+Dropout layer for each modality
        self.embed = dict()
        for m in self.modalities:
            self.embed[m] = nn.Sequential(nn.Dropout(0.1),
                                          nn.Linear(self.dims[m], embed_dim),
                                          nn.ReLU())
            self.add_module('embed_{}'.format(m), self.embed[m])
        # LSTM computes hidden states from embeddings and inter-event times
        self.lstm = nn.LSTM(1 + self.n_mods * embed_dim, h_dim,
                            n_layers, batch_first=True)
        # Regression network from LSTM hidden states to time-deltas
        self.time_out = nn.Sequential(nn.Linear(h_dim, embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(embed_dim, 1))
        # Regression network from LSTM hidden states to predicted value
        self.val_out = nn.Sequential(nn.Linear(h_dim, embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(embed_dim, 1))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, mask, lengths, estimate=False):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Convert NaNs to zeros so they don't affect embedding value
        for m in self.modalities:
            inputs[m][torch.isnan(inputs[m])] = 0
        # Convert raw features into equal-dimensional embeddings
        embed = torch.cat([self.embed[m](inputs[m].view(-1, self.dims[m]))
                           for m in self.modalities], dim=1)
        # Compute inter-event times, and concatenate to embeddings
        t_diff_in = inputs['time'] - pad_shift(inputs['time'], 1)
        embed = torch.cat([t_diff_in.view(-1, 1), embed], dim=1)
        # Unflatten temporal dimension
        embed = embed.reshape(batch_size, seq_len, -1)
        # Pack the input to mask padded entries
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        # Forward propagate LSTM
        h, _ = self.lstm(embed)
        # Undo the packing and flatten temporal dimension
        h, _ = pad_packed_sequence(h, batch_first=True)
        h = h.reshape(-1, self.h_dim)
        # Decode the hidden state to predicted time-deltas and values
        t_diff = self.time_out(h).view(batch_size, seq_len, 1)
        value = self.val_out(h).view(batch_size, seq_len, 1)
        # Mask entries that exceed sequence lengths
        t_diff = t_diff * mask.float()
        value = value * mask.float()
        if not estimate:
            # Return all predictions to compute loss
            return t_diff, value
        else:
            # Filter out out-of-order timestamps and return predict
            return self.estimate(inputs['time'], t_diff, value, mask)
            
    def estimate(self, time, t_diff, value, mask):
        # Sum to get next event times
        t_hat = time + t_diff
        # Iterate backwards and keep only prediced timestamps that are
        # smaller than the smallest timestamp seen so far
        t_min = t_hat[:,-1,:]
        keep = [mask[:,-1,:]]
        for i in reversed(range(0, t_hat.shape[1]-1)):
            k = mask[:,i,:] * (t_hat[:,i,:] < t_min)
            t_min = torch.min(t_min, t_hat[:,i,:])
            keep.append(k)
        keep = torch.cat(keep, dim=1).unsqueeze(-1)
        # Filter out late event times and predicted values
        t_filt, v_filt = [], []
        for seq_id in range(keep.shape[0]):
            t_seq, v_seq = t_hat[seq_id,:,:], value[seq_id,:,:]
            t_filt.append(t_seq[keep[seq_id,:,:]])
            v_filt.append(v_seq[keep[seq_id,:,:]])
        return t_filt, v_filt
            
    def loss(self, data, t_diff, value, mask, lambda_t=0.01):
        t_target, v_target = data['t_target'], data['v_target']
        time = data['time']
        # Find indices before last target is observed
        observed = (1 - torch.isnan(v_target)) * mask
        n_obs = observed.sum().item()
        if n_obs == 0:
            return torch.tensor(0.0).to(self.device), 0
        # Compute mean square loss for time-deltas
        loss = torch.sum(((t_target-time)[observed] - t_diff[observed])**2)
        loss *= lambda_t
        # Compute mean square loss between predicted values and targets
        loss += torch.sum((v_target[observed] - value[observed])**2)
        # Divide loss by number of non-padding datapoints
        n_obs = observed.sum().item()
        loss /= n_obs
        return loss, n_obs
    
if __name__ == "__main__":
    # Test code by loading dataset and running through model
    import os, argparse
    from datasets import load_dataset, seq_collate_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../../data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="Train",
                        help='whether to load Train/Valid/Test data')
    args = parser.parse_args()

    print("Loading data...")
    dataset = load_dataset(['acoustic', 'emotient', 'ratings'],
                           args.dir, args.subset, item_as_dict=True)
    print("Building model...")
    model = SemiParamNPP(['acoustic', 'emotient'], [988, 31],
                         device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths = seq_collate_dict([dataset[0]])
    times, preds = model(data, mask, lengths, estimate=True)
    print("Predicted valences:")
    for t, p in zip(list(times[0]), list(preds[0])):
        print("{:0.2f}, {:+0.3f}".format(t, p))
