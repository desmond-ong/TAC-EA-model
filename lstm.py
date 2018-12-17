from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MultiLSTM(nn.Module):
    """Multimodal LSTM model with feature level fusion.

    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    hidden_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    attn_len -- length of local attention window
    """
    
    def __init__(self, modalities, dims, embed_dim=128, hidden_dim=512,
                 n_layers=1, attn_len=3, device=torch.device('cuda:0')):
        super(MultiLSTM, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attn_len = attn_len
        
        # Create raw-to-embed FC+Dropout layer for each modality
        self.embed = dict()
        for m in self.modalities:
            self.embed[m] = nn.Sequential(nn.Dropout(),
                                          nn.Linear(self.dims[m], embed_dim),
                                          nn.ReLU())
            self.add_module('embed_{}'.format(m), self.embed[m])
        # Layer that computes attention from embeddings
        self.attn = nn.Sequential(nn.Linear(self.n_mods*embed_dim, embed_dim),
                                  nn.ReLU(),
                                  nn.Linear(embed_dim, attn_len),
                                  nn.Softmax(dim=1))
        # LSTM computes hidden states from embeddings for each modality
        self.lstm = nn.LSTM(self.n_mods*embed_dim, hidden_dim,
                            n_layers, batch_first=True)
        # Regression network from LSTM hidden states to predicted valence
        self.dec_target = nn.Sequential(nn.Linear(hidden_dim, embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(embed_dim, 1))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, mask, lengths, output_features=False):
        # Get batch dim
        batch_size, seq_len = len(lengths), max(lengths)
        # Convert raw features into equal-dimensional embeddings
        embed = torch.cat([self.embed[m](inputs[m].view(-1, self.dims[m]))
                           for m in self.modalities], 1)
        # Compute attention weights
        attn = self.attn(embed)
        # Unflatten temporal dimension
        embed = embed.reshape(batch_size, seq_len, self.n_mods*self.embed_dim)
        attn = attn.reshape(batch_size, seq_len, self.attn_len)
        # Pack the input to mask padded entries
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.n_layers, batch_size,
                         self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, batch_size,
                         self.hidden_dim).to(self.device)
        # Forward propagate LSTM
        h, _ = self.lstm(embed, (h0, c0))
        # Undo the packing
        h, _ = pad_packed_sequence(h, batch_first=True)
        # Convolve output with attention weights
        # i.e. out[t] = a[t,0]*in[t] + ... + a[t,win_len-1]*in[t-(win_len-1)]
        stacked = torch.stack([self.pad_shift(h, i) for
                               i in range(self.attn_len)], dim=-1)
        context = torch.sum(attn.unsqueeze(2) * stacked, dim=-1)
        # Flatten temporal dimension
        context = context.reshape(-1, self.hidden_dim)
        # Return features before final FC layer if flag is set
        if output_features:
            features = self.dec_target[0](context)
            features = features.view(batch_size, seq_len, -1) * mask.float()
            return features
        # Decode the context for each time step
        target = self.dec_target(context).view(batch_size, seq_len, 1)
        # Mask target entries that exceed sequence lengths
        target = target * mask.float()
        return target

    def pad_shift(self, x, shift):
        """Shift 3D tensor forwards in time with zero padding."""
        if shift > 0:
            padding = torch.zeros(x.size(0), shift, x.size(2)).to(self.device)
            return torch.cat((padding, x[:, :-shift, :]), dim=1)
        elif shift < 0:
            padding = torch.zeros(x.size(0), -shift, x.size(2)).to(self.device)
            pad = torch.cat((x[:, -shift:, :], padding), dim=1)
            return pad
        else:
            return x
    
if __name__ == "__main__":
    # Test code by loading dataset and running through model
    import os, argparse
    from datasets import load_dataset, seq_collate_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="Train",
                        help='whether to load Train/Valid/Test data')
    args = parser.parse_args()

    print("Loading data...")
    dataset = load_dataset(['acoustic', 'emotient', 'ratings'],
                           args.dir, args.subset, truncate=True,
                           item_as_dict=True)
    print("Building model...")
    model = MultiLSTM(['acoustic', 'emotient'], [988, 31],
                      device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths = seq_collate_dict([dataset[0]])
    del data['ratings']
    out = model(data, mask, lengths).view(-1)
    print("Predicted valences:")
    for o in out:
        print("{:+0.3f}".format(o.item()))
