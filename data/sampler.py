import torch
from torch.utils.data import Sampler, DataLoader
import random
import numpy as np
import os
from itertools import cycle

import torch.distributed as dist


class RandomSequenceSampler(Sampler):
    def __init__(self, data_path, batch_size, window_size, start_end_idxs=None):
        super().__init__()
        
        self.batch_size = batch_size
        self.window_size = window_size

        if start_end_idxs is None:
            assert os.path.isfile(data_path + '/ep_start_end_ids.npy'), "ep_start_end_ids.npy not found in the data path"
            self.idx_file = np.load(data_path + '/ep_start_end_ids.npy')
            self.start_end_idxs = [list(item) for item in self.idx_file]
        else:
            self.start_end_idxs = start_end_idxs
        
        # Initialize the current batch start-end indexes
        self.current_batch_start_end_idxs = random.sample(self.start_end_idxs, batch_size)
        self.length = len([i for start, end in self.start_end_idxs for i in range(start, end)])

        self.counter = {}
        for start_end_pair in self.current_batch_start_end_idxs:
            self.counter[start_end_pair[0]] = 1

    def __iter__(self):
        num_batches = self.length // (self.batch_size * self.window_size)
        
        for _ in range(num_batches):
            batch_sequences = []
            for i, (start_idx, end_idx) in enumerate(self.current_batch_start_end_idxs):
                window_end_idx = start_idx + self.window_size
                if window_end_idx < end_idx:
                    batch_sequences.append([start_idx, window_end_idx])
                    self.current_batch_start_end_idxs[i] = window_end_idx, end_idx
                else:
                    batch_sequences.append([start_idx, end_idx])
                    # Sample a new sequence
                    new_start_end_idx = random.sample(self.start_end_idxs, 1)
                    start_idx, end_idx = new_start_end_idx[0]

                    if start_idx in self.counter.keys():
                        self.counter[start_idx] += 1
                    else:
                        self.counter[start_idx] = 1

                    self.current_batch_start_end_idxs[i] = (start_idx, end_idx)
            # Yield the batch of sequences
            yield batch_sequences

    def __len__(self):
        return 42

class SequenceSampler(Sampler):
    def __init__(self, data_path, batch_size, window_size, n_obs_steps=0, start_end_idxs=None):
        super().__init__()
        
        self.batch_size = batch_size
        self.window_size = window_size
        self.n_obs_steps = n_obs_steps

        if start_end_idxs is None:
            assert os.path.isfile(data_path + '/ep_start_end_ids.npy'), "ep_start_end_ids.npy not found in the data path"
            self.idx_file = np.load(data_path + '/ep_start_end_ids.npy')
            self.start_end_idxs = [list(item) for item in self.idx_file]
        else:
            self.start_end_idxs = start_end_idxs
        self.current_batch_start_end_idxs = self.start_end_idxs[:batch_size]
        self.length = len([i for start, end in self.start_end_idxs for i in range(start, end)])
        self.start_end_idxs = cycle(self.start_end_idxs[self.batch_size:])
        self.num_batches = self.length // (self.batch_size * self.window_size)

        self.counter = {}
        for start_end_pair in self.current_batch_start_end_idxs:
            self.counter[start_end_pair[0]] = 1

    def __iter__(self):
        
        for b in range(self.num_batches):
            batch_sequences = []
            for i, (start_idx, end_idx) in enumerate(self.current_batch_start_end_idxs):
                window_end_idx = start_idx + self.window_size
                if window_end_idx < end_idx-self.n_obs_steps:
                    batch_sequences.append([start_idx, window_end_idx])
                    self.current_batch_start_end_idxs[i] = window_end_idx, end_idx
                else:
                    batch_sequences.append([start_idx, end_idx-self.n_obs_steps])
                    # Get next sequence start end idxs
                    new_start_end_idx = next(self.start_end_idxs)
                    start_idx = new_start_end_idx[0]
                    end_idx = new_start_end_idx[1]

                    if start_idx in self.counter.keys():
                        self.counter[start_idx] += 1
                    else:
                        self.counter[start_idx] = 1

                    self.current_batch_start_end_idxs[i] = (start_idx, end_idx)
            # Yield the batch of sequences
            print(batch_sequences)
            yield batch_sequences

    def __len__(self):
        return self.length // (self.batch_size * self.window_size)
    

class DistributedSequenceSampler(Sampler):
    def __init__(self, data_path, batch_size, window_size, n_obs_steps=0, num_replicas=None, rank=None, start_end_idxs=None):
        super().__init__()

        self.batch_size = batch_size
        self.window_size = window_size
        self.n_obs_steps = n_obs_steps

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.num_replicas = num_replicas
        self.rank = rank
        print(self.rank)

        if start_end_idxs is None:
            assert os.path.isfile(data_path + '/ep_start_end_ids.npy'), "ep_start_end_ids.npy not found in the data path"
            self.idx_file = np.load(data_path + '/ep_start_end_ids.npy')
            self.start_end_idxs = [list(item) for item in self.idx_file]
        else:
            self.start_end_idxs = start_end_idxs
        
        self.length = len([i for start, end in self.start_end_idxs for i in range(start, end)])

        self.num_samples = self.length // self.num_replicas
        self.num_idx_pairs_rank = len(self.start_end_idxs) // self.num_replicas

        # Distribute start_end_idxs among replicas
        self.rank_chunk_start = self.rank * self.num_idx_pairs_rank
        self.rank_chunk_end = min((self.rank + 1) * self.num_idx_pairs_rank, len(self.start_end_idxs))
        print(self.rank_chunk_start, self.rank_chunk_end, "Rank {0} Start End Idx".format(self.rank))

        # Get the subset of start_end_idxs for this process
        self.start_end_idxs = self.start_end_idxs[self.rank_chunk_start:self.rank_chunk_end]

        self.current_batch_start_end_idxs = self.start_end_idxs[:batch_size]
        
        self.start_end_idxs_cycle = cycle(self.start_end_idxs[self.batch_size:])

        self.num_batches = self.num_samples // (self.batch_size * self.window_size)

    #def __iter__(self):
    #
    #    for b in range(self.num_batches):
    #        batch_sequences = []
    #        for i, (start_idx, end_idx) in enumerate(self.current_batch_start_end_idxs):
    #            window_end_idx = start_idx + self.window_size
    #            if window_end_idx < end_idx:
    #                batch_sequences.append([start_idx, window_end_idx])
    #                self.current_batch_start_end_idxs[i] = window_end_idx, end_idx
    #            else:
    #                batch_sequences.append([start_idx, end_idx])
    #                new_start_end_idx = next(self.start_end_idxs_cycle)
    #                self.current_batch_start_end_idxs[i] = new_start_end_idx
    #        
    #        yield batch_sequences

    def __iter__(self):
        
        for b in range(self.num_batches):
            batch_sequences = []
            for i, (start_idx, end_idx) in enumerate(self.current_batch_start_end_idxs):
                window_end_idx = start_idx + self.window_size
                if window_end_idx < end_idx-self.n_obs_steps:
                    batch_sequences.append([start_idx, window_end_idx])
                    self.current_batch_start_end_idxs[i] = window_end_idx, end_idx
                else:
                    batch_sequences.append([start_idx, end_idx-self.n_obs_steps])
                    # Get next sequence start end idxs
                    new_start_end_idx = next(self.start_end_idxs)
                    start_idx = new_start_end_idx[0]
                    end_idx = new_start_end_idx[1]

                    if start_idx in self.counter.keys():
                        self.counter[start_idx] += 1
                    else:
                        self.counter[start_idx] = 1

                    self.current_batch_start_end_idxs[i] = (start_idx, end_idx)
            # Yield the batch of sequences
            yield batch_sequences

    def __len__(self):
        return self.num_samples // (self.batch_size * self.window_size)

if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader, Dataset

    # Define a custom dataset
    class CustomDataset(Dataset):
        def __init__(self, data, seq_len=2):
            self.data = data
            self.seq_len = seq_len

        def __len__(self):
            return len(self.start_end_idxs)

        def __getitem__(self, idx):
            seq = []
            for i in idx:
                seq.append(self.data[i])
            if len(seq) < self.seq_len:
                seq.append(seq[-1])
            return seq
        
        def collate_fn(batch):
            torch.tensor(batch)
            return batch
    
    class DiffDataset(Dataset):
        def __init__(self, data, n_obs_steps=0):
            self.data = data
            self.n_obs_steps = n_obs_steps

        def __len__(self):
            return len(self.start_end_idxs)

        def __getitem__(self, idx):
            obs = []
            labels = []
            for i in range(idx[0], idx[1]):
                obs.append(self.data[i])
            for i in range(idx[1], idx[1] + self.n_obs_steps):
                labels.append(self.data[i])
            if len(obs) < self.n_obs_steps:
                for i in range(self.n_obs_steps - len(obs)):
                    obs.append(obs[-1])
            while len(labels) < self.n_obs_steps:
                labels.append(labels[-1])
            return (obs, labels)
        
        def collate_fn(batch):
            print(batch)
            obs = torch.tensor(batch[0][0])
            labels = torch.tensor(batch[0][1])
            return obs, labels
    

    # Define some example data
    start_end_idxs = [[i * 20, (i + 1) * 20] for i in range(30)]  # Create 100 sequences, each of length 5
    data = [i for start, end in start_end_idxs for i in range(start, end)]

    # Initialize the custom dataset
    dataset = DiffDataset(data, 3)

    # Initialize the SequenceSampler
    batch_size = 1
    window_size = 3
    n_obs_steps = 3
    sampler = SequenceSampler(start_end_idxs, batch_size, window_size, n_obs_steps, start_end_idxs)

    # Create a PyTorch DataLoader using the SequenceSampler
    data_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=DiffDataset.collate_fn)

    # Iterate over the DataLoader and print the batches
    for i, batch in enumerate(data_loader):
        #print(f"Batch {i + 1}: {batch}")
        pass

        

   
