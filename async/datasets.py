from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, re, copy, itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiseqDataset(Dataset):
    """Multimodal dataset for asynchronous sequential data."""
    
    def __init__(self, modalities, dirs, regex, preprocess, time_cols,
                 time_tol=0, item_as_dict=False):
        """Loads valence ratings and features for each modality.
        Missing values are stored as NaNs.

        modalities -- names of each input modality
        dirs -- list of directories containing input features
        regex -- regex patterns for the filenames of each modality
        preprocess -- data pre-processing functions for pandas dataframes
        time_cols -- timestamp column names
        time_tol -- round timestamps to multiple of this number
        item_as_dict -- whether to return data as dictionary
        """
        # Store arguments
        self.modalities = modalities
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
        time_cols = {m: c for m, c in zip(modalities, time_cols)}
        
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
            
        # Load data from files
        self.data = []
        self.lengths = []
        self.cols = {}
        for i in range(len(self.seq_ids)):
            df = pd.DataFrame(columns=['time'])
            # Load each input modality
            for m in modalities:
                fp = paths[m][i]
                if re.match("^.*\.npy", fp):
                    d = pd.DataFrame(np.load(fp))
                elif re.match("^.*\.(csv|txt)", fp):
                    d = pd.read_csv(fp)
                elif re.match("^.*\.tsv", fp):
                    d = pd.read_csv(fp, sep='\t')
                # Preprocess timestamps
                d = d.rename(columns={time_cols[m]: 'time'})
                if time_tol > 0:
                    d['time'] = (d['time'] // time_tol) * time_tol
                # Preprocess non-time data
                t = d['time']
                d = preprocess[m](d.drop(columns=['time']))
                self.cols[m] = list(d.columns)
                d = pd.concat([t, d], axis=1)
                # Merge dataframe with previous modalities
                df = df.merge(d, how='outer', on='time')
            self.data.append(df)
            self.lengths.append(len(df))
            
    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, i):
        if self.item_as_dict:
            d = {m: self.data[i][self.cols[m]].values
                 for m in self.modalities}
            d['time'] = self.data[i][['time']].values
            d['length'] = self.lengths[i]
            return d
        else:
            d = [self.data[i][self.cols[m]].values for m in self.modalities]
            d = [self.data[i][['time']].values] + d
            return tuple(d)
        
    def normalize_(self):
        """Rescale all inputs to [-1, 1] range (in-place)."""
        # Find max and min for each dimension of each modality
        cols = self.cols
        m_max = {m: np.stack([d[cols[m]].max(0) for d in self.data]).max(0)
                 for m in self.modalities}
        m_min = {m: np.stack([d[cols[m]].min(0) for d in self.data]).min(0)
                 for m in self.modalities}
        # Compute range per dim and add constant to ensure it is non-zero
        m_rng = {m: (m_max[m]-m_min[m]) for m in self.modalities}
        m_rng = {m: m_rng[m] * (m_rng[m] > 0) + 1e-10 * (m_rng[m] <= 0)
                 for m in self.modalities}
        # Actually rescale the data
        for d in self.data:
            for m in self.modalities:
                d[cols[m]] = (d[cols[m]] - m_min[m]) / m_rng[m] * 2 - 1

    def normalize(self):
        """Rescale all inputs to [-1, 1] range (returns new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.normalize_()
        return dataset
            
    def split_(self, n):
        """Splits each sequence into n chunks (in place)."""
        self.data = list(itertools.chain.from_iterable(
            [np.array_split(d, n) for d in self.data]))
        self.seq_ids = list(itertools.chain.from_iterable(
            [[i] * n for i in self.seq_ids]))
        self.lengths = [len(d) for d in self.data[self.modalities[0]]]

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
        merged = copy.deepcopy(set1)
        merged.seq_ids += set2.seq_ids
        merged.data += copy.deepcopy(set2.data)
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
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data = zip(*data)
    lengths = [len(seq) for seq in data[0]]
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

def load_dataset(modalities, base_dir, subset,
                 time_tol=1.0/30, item_as_dict=False):
    """Helper function specifically for loading TAC-EA datasets."""
    dirs = {
        'acoustic': os.path.join(base_dir, 'features', subset, 'acoustic'),
        'linguistic': os.path.join(base_dir, 'features', subset, 'linguistic'),
        'emotient': os.path.join(base_dir, 'features', subset, 'emotient'),
        'ratings' : os.path.join(base_dir, 'ratings', subset, 'target')
    }
    regex = {
        'acoustic': "ID(\d+)_vid(\d+)_.*\.csv",
        'linguistic': "ID(\d+)_vid(\d+)_.*\.tsv",
        'emotient': "ID(\d+)_vid(\d+)_.*\.txt",
        'ratings' : "target_(\d+)_(\d+)_.*\.csv"
    }
    preprocess = {
        'acoustic': lambda df : df.drop(columns=['frameIndex']),
        'linguistic': lambda df : df.loc[:,'glove0':'glove299'],
        'emotient': lambda df : df,
        'ratings' : lambda df : df * 2 - 1
    }
    time_cols = {
        'acoustic': ' frameTime',
        'linguistic': 'time',
        'emotient': 'Frametime',
        'ratings' : 'time'
    }
    if 'ratings' not in modalities:
        modalities = modalities + ['ratings']
    return MultiseqDataset(modalities, [dirs[m] for m in modalities],
                           [regex[m] for m in modalities],
                           [preprocess[m] for m in modalities],
                           [time_cols[m] for m in modalities],
                           time_tol, item_as_dict)

if __name__ == "__main__":
    # Test code by loading dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="Train",
                        help='whether to load Train/Valid/Test data')
    args = parser.parse_args()

    print("Loading data...")
    modalities = ['acoustic', 'linguistic', 'emotient', 'ratings']
    dataset = load_dataset(modalities, args.dir, args.subset)
    print("Testing batch collation...")
    data = seq_collate([dataset[i] for i in range(min(10, len(dataset)))])
    print("Batch shapes:")
    for d in data[:-2]:
         print(d.shape)
    print("Sequence lengths: ", data[-1])
    print("Checking through data for mismatched sequence lengths...")
    for i, data in enumerate(dataset):
        print("Subject, Video: ", dataset.seq_ids[i])
        time, acoustic, linguistic, emotient, ratings = data
        print(acoustic.shape, linguistic.shape, emotient.shape, ratings.shape)
        if not (len(acoustic) == len(ratings) and
                len(linguistic) == len(ratings) and 
                len(emotient) == len(ratings)):
            print("WARNING: Mismatched sequence lengths.")
