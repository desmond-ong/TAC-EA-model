from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from scipy import integrate

def intensity_nll(t_diff, score, decay):
    log_intensity = score - decay * t_diff
    intensity = torch.exp(log_intensity)
    nll = -(log_intensity - 1/decay * (torch.exp(score) - intensity))
    return nll

def t_diff_mle(score, decay):
    f = lambda t : torch.exp(-intensity_nll(t, score, decay)).numpy()
    return integrate.quad(f, 0, np.inf)

class MultiNPP(nn.Module):
    """Multimodal neural point process (NPP) model.

    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    h_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    """
    
    def __init__(self, modalities, dims, embed_dim=128, h_dim=512,
                 n_layers=1, decay=0.1, device=torch.device('cuda:0')):
        super(MultiNPP, self).__init__()
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
        # LSTM computes hidden states from the summed embeddings
        self.lstm = nn.LSTM(self.n_mods * embed_dim, h_dim,
                            n_layers, batch_first=True)
        # Regression network from LSTM hidden states to predicted value
        self.val_out = nn.Sequential(nn.Linear(h_dim, embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(embed_dim, 1))
        # Network from LSTM hidden states to conditional intensity score
        self.time_out = nn.Linear(h_dim, 1)
        # Decay constant for intensity over time
        self.decay = nn.Parameter(torch.tensor(decay))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, mask, lengths, sample=False):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Convert NaNs to zeros so they don't affect embedding value
        for m in self.modalities:
            inputs[m][torch.isnan(inputs[m])] = 0
        # Convert raw features into equal-dimensional embeddings
        embed = torch.cat([self.embed[m](inputs[m].view(-1, self.dims[m]))
                           for m in self.modalities], dim=1)
        # Unflatten temporal dimension
        embed = embed.reshape(batch_size, seq_len, self.n_mods*self.embed_dim)
        # Pack the input to mask padded entries
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        # Forward propagate LSTM
        h, _ = self.lstm(embed)
        # Undo the packing and flatten temporal dimension
        h, _ = pad_packed_sequence(h, batch_first=True)
        h = h.reshape(-1, self.h_dim)
        # Decode the hidden state to event values and time intensity scores
        value = self.val_out(h).view(batch_size, seq_len, 1)
        score = self.time_out(h).view(batch_size, seq_len, 1)
        # Mask entries that exceed sequence lengths
        value = value * mask.float()
        score = score * mask.float()
        return value, score

    def loss(self, value, score, t_diff, target):
        # TODO: only compute loss for timestamps where target is observed
        loss = intensity_nll(t_diff, score, self.decay)
        loss += torch.sum((target - value)**2)
        return loss
    
if __name__ == "__main__":
    # Test code by loading dataset and running through model
    import os, argparse
    from datasets import load_dataset, seq_collate_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="Train",
                        help='whether to load Train/Valid/Test data')
    args = parser.parse_args()

    print("Loading data...")
    dataset = load_dataset(['acoustic', 'emotient', 'ratings'],
                           args.dir, args.subset, item_as_dict=True)
    print("Building model...")
    model = MultiNPP(['acoustic', 'emotient'], [988, 31],
                     device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths = seq_collate_dict([dataset[0]])
    out, score = model(data, mask, lengths)
    print("Predicted valences:")
    for o in out.view(-1):
        print("{:+0.3f}".format(o.item()))
