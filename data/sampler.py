import torch
from torch.utils.data import Sampler, DataLoader
import random
import numpy as np
import os

class SequenceSampler(Sampler):
    def __init__(self, data_path, batch_size, window_size):
        super().__init__()
        
        self.batch_size = batch_size
        self.window_size = window_size

        assert os.path.isfile(data_path + '/ep_start_end_ids.npy'), "ep_start_end_ids.npy not found in the data path"
        self.idx_file = np.load(data_path + '/ep_start_end_ids.npy')
        self.start_end_idxs = [list(item) for item in self.idx_file]
        
        # Initialize the current batch start-end indexes
        self.current_batch_start_end_idxs = random.sample(self.start_end_idxs, batch_size)
        self.length = len([i for start, end in self.start_end_idxs for i in range(start, end)])

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
                    self.current_batch_start_end_idxs[i] = (start_idx, end_idx)
            # Yield the batch of sequences
            yield batch_sequences

    def __len__(self):
        return self.length // (self.batch_size * self.window_size)

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

    # Define some example data
    start_end_idxs = [[i * 5, (i + 1) * 5] for i in range(100)]  # Create 100 sequences, each of length 5
    data = [i for start, end in start_end_idxs for i in range(start, end)]

    # Initialize the custom dataset
    dataset = CustomDataset(data, 3)

    # Initialize the SequenceSampler
    batch_size = 5
    window_size = 3
    sampler = SequenceSampler(start_end_idxs, batch_size, window_size)

    # Create a PyTorch DataLoader using the SequenceSampler
    data_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=CustomDataset.collate_fn)

    # Iterate over the DataLoader and print the batches
    for i, batch in enumerate(data_loader):
        print(f"Batch {i + 1}: {batch}")

        # Break the loop after 10 batches
        if i == 2:
            break

   
