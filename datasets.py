from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, re, copy, itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiseqDataset(Dataset):
    """Multimodal dataset for time series and sequential data."""
    
    def __init__(self, modalities=None, dirs=None, regex=None,
                 rates=None, base_rate=None, preprocess=None,
                 truncate=False, item_as_dict=False):
        """Loads valence ratings and features for each modality.

        modalities -- names of each input modality
        dirs -- list of directories containing input features
        regex -- regex patterns for the filenames of each modality
        rates -- sampling rates of each modality
        base_rate -- base_rate to subsample to
        preprocess -- data pre-processing functions for pandas dataframes
        truncate -- if true, truncate to modality with minimum length
        item_as_dict -- whether to return data as dictionary
        """
        # Store arguments
        self.modalities = modalities
        if type(rates) is not list:
            self.rates = [rates] * len(modalities)
        else:
            self.rates = rates
        self.base_rate = base_rate if base_rate != None else min(self.rates)
        self.item_as_dict = item_as_dict

        # Convert to modality-indexed dictionaries
        dirs = {m: d for m, d in zip(modalities, dirs)}
        if type(regex) is not list:
            regex = [regex] * len(self.modalities)
        regex = {m: r for m, r in zip(modalities, regex)}
        if preprocess is None:
            preprocess = lambda x : x
        if type(preprocess) is not list:
            preprocess = [preprocess] * len(self.modalities)
        preprocess = {m: p for m, p in zip(modalities, preprocess)}
        
        # Load filenames into lists and extract regex-captured sequence IDs
        paths = dict()
        seq_ids = dict()
        for m in modalities:
            paths[m] = []
            seq_ids[m] = []
            for fn in os.listdir(dirs[m]):
                match = re.match(regex[m], fn)
                if not match:
                    continue
                paths[m].append(os.path.join(dirs[m], fn))
                seq_ids[m].append(match.groups())
            # Sort by values of captured indices
            paths[m] = [p for _, p in sorted(zip(seq_ids[m], paths[m]))]
            seq_ids[m].sort()

        # Check that number and IDs of files/sequences are matched
        self.seq_ids = seq_ids[modalities[0]]
        for m in modalities:
            if len(paths[m]) != len(self.seq_ids):
                raise Exception("Number of files ({}) do not match.".\
                                format(len(paths[m])))
            if seq_ids[m] != self.seq_ids:
                raise Exception("Sequence IDs do not match.")

        # Compute ratio to base rate
        self.ratios = {m: int(r/self.base_rate) for m, r in
                       zip(self.modalities, self.rates)}
            
        # Load data from files
        self.data = {m: [] for m in modalities}
        self.lengths = []
        for i in range(len(self.seq_ids)):
            seq_len = float('inf')
            # Load each input modality
            for m, data in self.data.iteritems():
                fp = paths[m][i]
                if re.match("^.*\.(csv|txt)", fp):
                    # Use pandas to read and pre-process CSV files
                    d = pd.read_csv(fp)
                    d = np.array(preprocess[m](d))
                else:
                    # Otherwise load as numpy array
                    d = np.load(fp)
                # Flatten inputs
                if len(d.shape) > 2:
                    d = d.reshape(d.shape[0], -1)
                # Time average so that data is at base rate
                ratio = self.ratios[m]
                end = ratio * (len(d)//ratio)
                avg = np.mean(d[:end].reshape(-1, ratio, d.shape[1]), axis=1)
                if end < len(d):
                    remain = d[end:].mean(axis=0)[np.newaxis,:]
                    d = np.concatenate([avg, remain])
                else:
                    d = avg
                data.append(d)
                if len(d) < seq_len:
                    seq_len = len(d)
            # Truncate to minimum sequence length
            if truncate:
                for m in self.modalities:
                    self.data[m][-1] = self.data[m][-1][:seq_len]
            self.lengths.append(seq_len)
            
    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, i):
        if self.item_as_dict:
            d = {m: self.data[m][i] for m in self.modalities}
            d['length'] = self.lengths[i]
            return d
        else:
            return tuple(self.data[m][i] for m in self.modalities)
        
    def normalize_(self):
        """Rescale all inputs to [-1, 1] range (in-place)."""
        # Find max and min for each dimension of each modality
        m_max = {m: np.stack([a.max(0) for a in self.data[m]]).max(0)
                  for m in self.modalities}
        m_min = {m: np.stack([a.min(0) for a in self.data[m]]).min(0)
                  for m in self.modalities}
        # Compute range per dim and add constant to ensure it is non-zero
        m_rng = {m: (m_max[m]-m_min[m]) for m in self.modalities}
        m_rng = {m: m_rng[m] * (m_rng[m] > 0) + 1e-10 * (m_rng[m] <= 0)
                  for m in self.modalities}
        # Actually rescale the data
        for m in self.modalities:
            self.data[m] = [(a-m_min[m]) / m_rng[m] * 2 - 1 for
                               a in self.data[m]]

    def normalize(self):
        """Rescale all inputs to [-1, 1] range (returns new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.normalize_()
        return dataset
            
    def split_(self, n):
        """Splits each sequence into n chunks (in place)."""
        for m in self.modalities:
            self.seq_ids = list(itertools.chain.from_iterable(
                [[i] * n for i in self.seq_ids]))
            self.data[m] = list(itertools.chain.from_iterable(
                [np.array_split(a, n, 0) for a in self.data[m]]))

    def split(self, n):
        """Splits each sequence into n chunks (returns new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.split_(n)
        return dataset
            
    @classmethod
    def merge(cls, set1, set2):
        """Merge two datasets."""
        if (set1.modalities != set2.modalities):
            raise Exception("Modalities need to match.")
        if (set1.base_rate != set2.base_rate):
            raise Exception("Base rates need to match.")
        merged = copy.deepcopy(set1)
        merged.seq_ids += set2.seq_ids
        merged.rates = [merged.base_rate] * len(merged.modalities)
        for m in merged.modalities:
            merged.data[m] += set2.data[m]
        return merged
        
def len_to_mask(lengths):
    """Converts list of sequence lengths to a mask tensor."""
    mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths))
    mask = mask < torch.tensor(lengths).unsqueeze(1)
    return mask.unsqueeze(-1)

def pad_and_merge(sequences, max_len=None):
    """Pads and merges unequal length sequences into batch tensor."""
    dims = sequences[0].shape[1]
    lengths = [len(seq) for seq in sequences]
    if max_len is None:
        max_len = max(lengths)
    padded_seqs = torch.zeros(len(sequences), max_len, dims)
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end, :] = torch.from_numpy(seq[:end,:])
    if len(sequences) == 1:
        padded_seqs = padded_seqs.float()
    return padded_seqs

def seq_collate(data):
    """Collates multimodal variable length sequences into padded batch."""
    padded = []
    n_modalities = len(data)
    lengths = np.zeros(n_modalities, dtype=int)
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data = zip(*data)
    for modality in data:
        m_lengths = [len(seq) for seq in modality]
        lengths = np.maximum(lengths, m_lengths)
    lengths = list(lengths)
    for modality in data:
        padded.append(pad_and_merge(modality, max(lengths)))
    mask = len_to_mask(lengths)
    return tuple(padded + [mask, lengths])

def seq_collate_dict(data):
    """Collate that accepts and returns dictionaries."""
    batch = {}
    modalities = [k for k in data[0].keys() if  k != 'length']
    data.sort(key=lambda d: d['length'], reverse=True)
    lengths = [d['length'] for d in data]
    for m in modalities:
        m_data = [d[m] for d in data]
        batch[m] = pad_and_merge(m_data, max(lengths))
    mask = len_to_mask(lengths)
    return batch, mask, lengths

if __name__ == "__main__":
    # Test code by loading dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="Train",
                        help='whether to load Train/Valid/Test data')
    args = parser.parse_args()

    modalities = ['acoustic', 'emotient', 'ratings']
    dirs = [
        os.path.join(args.dir, 'features', args.subset, 'acoustic'),
        os.path.join(args.dir, 'features', args.subset, 'emotient'),
        os.path.join(args.dir, 'ratings', args.subset, 'target')
    ]
    regex = [
        "ID(\d+)_vid(\d+)_.*\.csv",
        "ID(\d+)_vid(\d+)_.*\.txt",
        "target_(\d+)_(\d+)_.*\.csv"
    ]
    rates = [2, 30, 2]
    preprocess = [
        lambda df : df.drop(columns=['frameIndex', ' frameTime']),
        lambda df : df.drop(columns=['Frametime']),
        lambda df : df.drop(columns=['time'])
    ]
    
    print("Loading data...")
    dataset = MultiseqDataset(modalities, dirs, regex, rates,
                              preprocess=preprocess)
    print("Testing batch collation...")
    data = seq_collate([dataset[i] for i in range(min(10, len(dataset)))])
    print("Batch shapes:")
    for d in data[:-1]:
        print(d.shape)
    print("Sequence lengths: ", data[-1])
    print("Checking through data for mismatched sequence lengths...")
    for i, data in enumerate(dataset):
        print("Subject, Video: ", dataset.seq_ids[i])
        acoustic, emotient, ratings = data
        print(acoustic.shape, emotient.shape, ratings.shape)
        if not (len(acoustic) == len(emotient) and
                len(emotient) == len(ratings)):
            print("WARNING: Mismatched sequence lengths.")
