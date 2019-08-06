from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, re, copy, itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiseqDataset(Dataset):
    """Multimodal dataset for (synchronous) time series and sequential data."""
    
    def __init__(self, modalities, dirs, regex, preprocess, rates,
                 base_rate=None, truncate=False, item_as_dict=False):
        """Loads valence ratings and features for each modality.

        modalities -- names of each input modality
        dirs -- list of directories containing input features
        regex -- regex patterns for the filenames of each modality
        preprocess -- data pre-processing functions for pandas dataframes
        rates -- sampling rates of each modality
        base_rate -- base_rate to subsample/ovesample to
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
        self.ratios = {m: r/self.base_rate for m, r in
                       zip(self.modalities, self.rates)}
            
        # Load data from files
        self.data = {m: [] for m in modalities}
        self.orig = {m: [] for m in modalities}
        self.lengths = []
        for i in range(len(self.seq_ids)):
            seq_len = float('inf')
            # Load each input modality
            for m, data in self.data.iteritems():
                fp = paths[m][i]
                if re.match("^.*\.npy", fp):
                    # Load as numpy array
                    d = np.load(fp)
                elif re.match("^.*\.(csv|txt)", fp):
                    # Use pandas to read and pre-process CSV files
                    d = pd.read_csv(fp)
                    d = np.array(preprocess[m](d))
                elif re.match("^.*\.tsv", fp):
                    d = pd.read_csv(fp, sep='\t')
                    d = np.array(preprocess[m](d))
                # Flatten inputs
                if len(d.shape) > 2:
                    d = d.reshape(d.shape[0], -1)
                # Store original data before resampling
                self.orig[m].append(d)
                # Subsample/oversample datat base rate
                ratio = self.ratios[m]
                if ratio > 1:
                    # Time average so that data is at base rate
                    ratio = int(ratio)
                    end = ratio * (len(d)//ratio)
                    avg = np.mean(d[:end].reshape(-1, ratio, d.shape[1]), 1)
                    if end < len(d):
                        remain = d[end:].mean(axis=0)[np.newaxis,:]
                        d = np.concatenate([avg, remain])
                    else:
                        d = avg
                else:
                    # Repeat so that data is at base rate
                    ratio = int(1. / ratio)
                    d = np.repeat(d, ratio, axis=0)
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

    def mean_and_std(self, modalities=None):
        """Compute mean+std across time and samples for given modalities."""
        if modalities is None:
            modalities = self.modalities
        m_mean = {m: np.nanmean(np.concatenate(self.data[m], 0), axis=0)
                  for m in modalities}
        m_std = {m: np.nanstd(np.concatenate(self.data[m], 0), axis=0)
                 for m in modalities}
        return m_mean, m_std

    def max_and_min(self, modalities=None):
        """Compute max+min across time and samples for given modalities."""
        if modalities is None:
            modalities = self.modalities
        m_max = {m: np.nanmax(np.stack([a.max(0) for a in self.data[m]]), 0)
                 for m in modalities}
        m_min = {m: np.nanmin(np.stack([a.min(0) for a in self.data[m]]), 0)
                 for m in modalities}
        return m_max, m_min
    
    def normalize_(self, modalities=None, method='meanvar', ref_data=None):
        """Normalize data either by mean-and-variance or to [-1,1])."""
        if modalities is None:
            # Default to all modalities
            modalities = self.modalities
        if ref_data is None:
            # Default to computing stats over self
            ref_data = self
        if method == 'range':
            # Range normalization
            m_max, m_min = ref_data.max_and_min(modalities)
            # Compute range per dim and add constant to ensure it is non-zero
            m_rng = {m: (m_max[m]-m_min[m]) for m in modalities}
            m_rng = {m: m_rng[m] * (m_rng[m] > 0) + 1e-10 * (m_rng[m] <= 0)
                     for m in modalities}
            for m in modalities:
                self.data[m] = [(a-m_min[m]) / m_rng[m] * 2 - 1 for
                                a in self.data[m]]
        else:
            # Mean-variance normalization
            m_mean, m_std = ref_data.mean_and_std(modalities)
            for m in modalities:
                self.data[m] = [(a-m_mean[m]) / (m_std[m] + 1e-10) for
                                a in self.data[m]]

    def normalize(self, modalities=None, method='meanvar', ref_data=None):
        """Normalize data (returns new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.normalize_(modalities, method, ref_data)
        return dataset
            
    def split_(self, n):
        """Splits each sequence into n chunks (in place)."""
        for m in self.modalities:
            self.data[m] = list(itertools.chain.from_iterable(
                [np.array_split(a, n, 0) for a in self.data[m]]))
        self.seq_ids = list(itertools.chain.from_iterable(
            [[i] * n for i in self.seq_ids]))
        self.lengths = [len(d) for d in self.data[self.modalities[0]]]

    def split(self, n):
        """Splits each sequence into n chunks (returns new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.split_(n)
        return dataset

    def unravel_(self, mod):
        """Unravels columns in specified modality, creating a new 
           sequence for each column."""
        new_data = {m: [] for m in self.modalities}
        new_seq_ids = []
        for i in range(len(self.seq_ids)):
            n_cols = self.data[mod][i].shape[1]
            for col in range(n_cols):
                for m in self.modalities:
                    if m == mod:
                        # Select column from specified modality
                        new_data[mod].append(self.data[mod][i][:,col:col+1])
                    else:
                        # Copy data for other modalities
                        new_data[m].append(self.data[m][i])
            new_seq_ids += n_cols * [self.seq_ids[i]]
        self.data = new_data
        self.seq_ids = new_seq_ids
        self.lengths = [len(d) for d in self.data[mod]]
            
    def unravel(self, mod):
        """Unravels columns in specified modality (return new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.unravel_(mod)
        return dataset
            
    @classmethod
    def merge(cls, set1, set2):
        """Merge two datasets."""
        if (set1.modalities != set2.modalities):
            raise Exception("Modalities need to match.")
        if (set1.base_rate != set2.base_rate):
            raise Exception("Base rates need to match.")
        merged = copy.deepcopy(set1)
        merged.orig.clear()
        merged.seq_ids += set2.seq_ids
        merged.rates = [merged.base_rate] * len(merged.modalities)
        merged.ratios = [1] * len(merged.modalities)
        for m in merged.modalities:
            merged.data[m] += copy.deepcopy(set2.data[m])
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

def seq_collate(data, time_first=True):
    """Collates multimodal variable length sequences into padded batch."""
    padded = []
    n_modalities = len(data) #n_modalities = len(data[0])
    lengths = np.zeros(n_modalities, dtype=int)
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data = zip(*data)
    for modality in data:
        m_lengths = [len(seq) for seq in modality]
        lengths = np.maximum(lengths, m_lengths)
    lengths = list(lengths)
    for modality in data:
        m_padded = pad_and_merge(modality, max(lengths))
        padded.append(m_padded.permute(1, 0, 2) if time_first else m_padded)
    mask = len_to_mask(lengths)
    if time_first:
        mask = mask.permute(1, 0, 2)
    return tuple(padded + [mask, lengths])

def seq_collate_dict(data, time_first=True):
    """Collate that accepts and returns dictionaries."""
    batch = {}
    modalities = [k for k in data[0].keys() if  k != 'length']
    data.sort(key=lambda d: d['length'], reverse=True)
    lengths = [d['length'] for d in data]
    for m in modalities:
        m_data = [d[m] for d in data]
        m_padded = pad_and_merge(m_data, max(lengths))
        batch[m] = m_padded.permute(1, 0, 2) if time_first else m_padded
    mask = len_to_mask(lengths)
    if time_first:
        mask = mask.permute(1, 0, 2)
    return batch, mask, lengths

def load_dataset(modalities, base_dir, subset,
                 base_rate=None, truncate=False, item_as_dict=False,
                 ratings_info=None):
    """Helper function specifically for loading TAC-EA datasets."""
    dirs = {
        'acoustic': os.path.join(base_dir, 'features',
                                 subset, 'acoustic-egemaps'),
        'linguistic': os.path.join(base_dir, 'features',
                                   subset, 'linguistic'),
        'emotient': os.path.join(base_dir, 'features',
                                 subset, 'emotient'),
        'ratings' : os.path.join(base_dir, 'ratings',
                                 subset, 'observer_EWE')
        # 'ratings' : os.path.join(base_dir, 'ratings', subset, 'target')
    }
    regex = {
        'acoustic': "ID(\d+)_vid(\d+)_.*\.csv",
        'linguistic': "ID(\d+)_vid(\d+)_.*\.tsv",
        'emotient': "ID(\d+)_vid(\d+)_.*\.txt",
        'ratings' : "results_(\d+)_(\d+)\.csv" #observer_EWE
        # 'ratings' : "target_(\d+)_(\d+)_normal\.csv" #target
    }
    rates = {'acoustic': 2, 'linguistic': 0.2, 'emotient': 30, 'ratings': 2}
    preprocess = {
        # Drop timestamps and frame indices
        'acoustic': lambda df : df.drop(columns=['name', ' frameTime']),
        # Use only GloVe vectors, leave missing as NaN
        'linguistic': lambda df : df.loc[:,'glove0':'glove299'],
        # Fill in missing emotient data with NaNs, use only action units
        'emotient': lambda df : (df.set_index('Frametime')\
                                 .reindex(
                                     np.arange(0.0333667,
                                               max(df['Frametime']),
                                               0.0333667),
                                     axis='index', method='nearest',
                                     tolerance=1e-3, fill_value=float('nan'))\
                                 .reset_index().loc[:,'AU1':'AU43']),
        # Rescale from [0, 100] to [-1, 1]
        'ratings' : lambda df : df.loc[:,['evaluatorWeightedEstimate']]/50 - 1
        # 'ratings' : lambda df : df.drop(columns=['time']) * 2 - 1 #target
    }
    if 'ratings' not in modalities:
        modalities = modalities + ['ratings']
    # Override load options for ratings, if provided
    if ratings_info is not None:
        if 'dirs' in ratings_info:
            dirs['ratings'] = ratings_info['dirs']
        if 'regex' in ratings_info:
            regex['ratings'] = ratings_info['regex']
        if 'rates' in ratings_info:
            rates['ratings'] = ratings_info['rates']
        if 'preprocess' in ratings_info:
            preprocess['ratings'] = ratings_info['preprocess']
    return MultiseqDataset(modalities, [dirs[m] for m in modalities],
                           [regex[m] for m in modalities],
                           [preprocess[m] for m in modalities],
                           [rates[m] for m in modalities],
                           base_rate, truncate, item_as_dict)

def load_dataset_multi(modalities, base_dir, subset,
                       base_rate=None, truncate=False, item_as_dict=False):    
    """Helper function to load dataset with multiple observer ratings."""
    ratings_info = {
        'dirs' : os.path.join(base_dir, 'ratings', subset, 'observer'),
        'regex' : "results_(\d+)_(\d+)\.csv",
        'preprocess' : lambda df : df.drop(columns=['workerID']) / 50 - 1
    }
    dataset = load_dataset(modalities, base_dir, subset,
                           base_rate, truncate, item_as_dict, ratings_info)
    # Unravel ratings columns so that each observer has one sequence
    dataset.unravel_('ratings')
    return dataset

if __name__ == "__main__":
    # Test code by loading dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../../data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="Train",
                        help='whether to load Train/Valid/Test data')
    parser.add_argument('--modalities', type=str, default=None, nargs='+',
                        help='input modalities (default: all')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='whether to compute and print statistics')
    args = parser.parse_args()

    print("Loading data...")
    if args.modalities is None:
        modalities = ['acoustic', 'linguistic', 'emotient', 'ratings']
    else:
        modalities = args.modalities
    dataset = load_dataset(modalities, args.dir, args.subset, base_rate=2.0)
    print("Testing batch collation...")
    data = seq_collate([dataset[i] for i in range(min(10, len(dataset)))])
    print("Batch shapes:")
    for d in data[:-2]:
        print(d.shape)
    print("Sequence lengths: ", data[-1])
    print("Checking through data for mismatched sequence lengths...")
    for i, data in enumerate(dataset):
        print("Subject, Video: ", dataset.seq_ids[i])
        acoustic, linguistic, emotient, ratings = data
        print(acoustic.shape, linguistic.shape, emotient.shape, ratings.shape)
        if not (len(acoustic) == len(ratings) and
                len(linguistic) == len(ratings) and 
                len(emotient) == len(ratings)):
            print("WARNING: Mismatched sequence lengths.")

    if args.stats:
        print("Statistics:")
        m_mean, m_std = dataset.mean_and_std()
        m_max, m_min = dataset.max_and_min()
        for m in modalities:
            print("--", m, "--")
            print("Mean:", m_mean[m])
            print("Std:", m_std[m])
            print("Max:", m_max[m])
            print("Min:", m_min[m])
