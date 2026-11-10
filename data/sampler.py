import torch
from torch.utils.data import Sampler, DataLoader, BatchSampler
import random
import numpy as np
import os
import math
from itertools import cycle

import torch.distributed as dist
from copy import deepcopy


class ResumableBatchSampler(Sampler):
    """Batch indices normally, but allow cheap mid-epoch index skipping."""

    def __init__(self, sampler, batch_size, drop_last):
        super().__init__()
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.start_batch = 0

    def set_start_batch(self, start_batch):
        start_batch = int(start_batch)
        if not 0 <= start_batch < len(self.batch_sampler):
            raise ValueError(
                f"start_batch must be in [0, {len(self.batch_sampler)}), "
                f"got {start_batch}"
            )
        self.start_batch = start_batch

    def __iter__(self):
        for batch_index, batch in enumerate(self.batch_sampler):
            if batch_index >= self.start_batch:
                yield batch

    def __len__(self):
        return len(self.batch_sampler) - self.start_batch


class ManifestSequenceBatchSampler(Sampler):
    """Yield fixed-length chunks while preserving episode order in each lane.

    Every item emitted to the dataset is ``[start, end_exclusive]``.  A lane
    keeps consuming consecutive chunks from one episode and only switches to
    another episode at a boundary, allowing the model's recurrent state mask
    to reset exactly at manifest-selected episode starts.

    Training may request a fixed number of batches so Episodic and
    Non-Episodic factorial cells have the same optimizer-step budget.  For
    validation, ``num_batches`` is omitted and the episode-to-lane assignment
    is balanced so every selected episode is traversed at least once, with
    only the shorter lanes repeating enough chunks to fill rectangular
    batches.
    """

    def __init__(
        self,
        start_end_idxs,
        batch_size,
        window_size,
        *,
        seed,
        shuffle,
        num_batches=None,
        num_replicas=1,
        rank=0,
    ):
        super().__init__()
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for {num_replicas} replicas")
        self.batch_size = int(batch_size)
        self.window_size = int(window_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.start_batch = 0

        all_ranges = [tuple(map(int, item)) for item in start_end_idxs]
        self.start_end_idxs = all_ranges[rank::num_replicas]
        if len(self.start_end_idxs) < self.batch_size:
            raise ValueError(
                f"Rank {rank} has {len(self.start_end_idxs)} episodes, fewer than "
                f"batch size {self.batch_size}"
            )
        for start, end in self.start_end_idxs:
            if end < start:
                raise ValueError(f"Invalid inclusive episode range: {(start, end)}")

        self._requested_num_batches = (
            None if num_batches is None else int(num_batches)
        )
        if self._requested_num_batches is not None and self._requested_num_batches <= 0:
            raise ValueError("num_batches must be positive")
        self.num_batches = 0
        self.num_samples = 0
        self.natural_num_chunks = sum(
            self._chunk_count(item) for item in self.start_end_idxs
        )
        self._refresh_lengths()

    def _chunk_count(self, episode_range):
        start, end = episode_range
        return math.ceil((end - start + 1) / self.window_size)

    def _build_lanes(self):
        episodes = list(self.start_end_idxs)
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(episodes)
        # Stable sorting after the seeded shuffle gives balanced lane lengths
        # while retaining seeded tie-breaking among similarly sized episodes.
        episodes.sort(key=self._chunk_count, reverse=True)
        lanes = [[] for _ in range(self.batch_size)]
        lane_loads = [0] * self.batch_size
        for episode in episodes:
            lane = min(range(self.batch_size), key=lane_loads.__getitem__)
            lanes[lane].append(episode)
            lane_loads[lane] += self._chunk_count(episode)
        if any(not lane for lane in lanes):
            raise ValueError("Every sequence lane must contain at least one episode")
        return lanes, lane_loads

    def _refresh_lengths(self):
        _, lane_loads = self._build_lanes()
        natural_batches = max(lane_loads)
        self.num_batches = self._requested_num_batches or natural_batches
        self.num_samples = self.num_batches * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.start_batch = 0
        self._refresh_lengths()

    def set_start_batch(self, start_batch):
        start_batch = int(start_batch)
        if not 0 <= start_batch < self.num_batches:
            raise ValueError(
                f"start_batch must be in [0, {self.num_batches}), got {start_batch}"
            )
        self.start_batch = start_batch

    def __iter__(self):
        lanes, _ = self._build_lanes()
        episode_positions = [0] * self.batch_size
        frame_positions = [lane[0][0] for lane in lanes]

        for batch_index in range(self.num_batches):
            batch = []
            for lane_index, lane in enumerate(lanes):
                episode_index = episode_positions[lane_index]
                start, end_inclusive = lane[episode_index]
                current = frame_positions[lane_index]
                end_exclusive = min(current + self.window_size, end_inclusive + 1)
                batch.append([current, end_exclusive])

                if end_exclusive >= end_inclusive + 1:
                    episode_index = (episode_index + 1) % len(lane)
                    episode_positions[lane_index] = episode_index
                    frame_positions[lane_index] = lane[episode_index][0]
                else:
                    frame_positions[lane_index] = end_exclusive
            # Advancing the lane state is cheap and deterministic.  Do not
            # yield skipped batches, so DataLoader workers never read their
            # image/NPZ data during a mid-epoch resume.
            if batch_index >= self.start_batch:
                yield batch

    def __len__(self):
        return self.num_batches - self.start_batch


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
    
class DistributedRandomSampler(Sampler):
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
            self.start_end_idxs = list(start_end_idxs)
        
        # compute exact number of batches
        self.length = len([i for start, end in self.start_end_idxs for i in range(start, end)])
        print(self.length, "Length")
        num_windows = 0
        for start_end_idx in self.start_end_idxs:
            num_windows += (start_end_idx[1] - start_end_idx[0]) // self.window_size + 1
        self.num_samples = num_windows // self.num_replicas

        self.num_idx_pairs_rank = len(self.start_end_idxs) // self.num_replicas

        # Distribute start_end_idxs among replicas
        self.rank_chunk_start = self.rank * self.num_idx_pairs_rank
        self.rank_chunk_end = min((self.rank + 1) * self.num_idx_pairs_rank, len(self.start_end_idxs))
        print(self.rank_chunk_start, self.rank_chunk_end, "Rank {0} Start End Idx".format(self.rank))

        # Get the subset of start_end_idxs for this process
        self.start_end_idxs = self.start_end_idxs[self.rank_chunk_start:self.rank_chunk_end]

        self.current_batch_start_end_idxs = self.start_end_idxs[:batch_size]
        
        self.available_start_end_idxs = self.start_end_idxs[self.batch_size:]
        
        self.num_batches = self.num_samples // self.batch_size
        print(self.num_batches, "Num Batches")
        self.counter = {}

    def __iter__(self):
        for b in range(self.num_batches):
            batch_sequences = []
            for i, (start_idx, end_idx) in enumerate(self.current_batch_start_end_idxs):
                window_end_idx = start_idx + self.window_size
                if window_end_idx < end_idx - self.n_obs_steps:
                    batch_sequences.append([start_idx, window_end_idx])
                    self.current_batch_start_end_idxs[i] = window_end_idx, end_idx
                else:
                    batch_sequences.append([start_idx, end_idx - self.n_obs_steps])
                    # Get next sequence start end idxs
                    if len(self.available_start_end_idxs) == 0:
                        self.available_start_end_idxs = deepcopy(self.start_end_idxs)
                    new_start_end_idx = random.choice(self.available_start_end_idxs)
                    self.available_start_end_idxs.remove(new_start_end_idx)
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
        return self.num_samples // self.batch_size

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
            for i in range(idx[0], idx[1]):
                seq.append(self.data[i])
            if len(seq) < self.seq_len:
                seq.append(seq[-1])
            return seq
        
        def collate_fn(batch):
            torch.tensor(batch)
            return batch

    # Define some example data
    start_end_idxs = [[i * 5, (i+1) * 5] for i in range(20)]  # Create 100 sequences, each of length 5
    data = [i for start, end in start_end_idxs for i in range(start, end)]

    # Initialize the custom dataset
    dataset = CustomDataset(data, 3)

    # Initialize the SequenceSampler
    batch_size = 5
    window_size = 2
    sampler = DistributedRandomSampler(data_path='.', batch_size=batch_size, window_size=window_size, start_end_idxs=start_end_idxs, num_replicas=1, rank=0)

    # Create a PyTorch DataLoader using the SequenceSampler
    data_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=CustomDataset.collate_fn)

    # Iterate over the DataLoader and print the batches
    for e in range(5):
        print(f"Epoch {e + 1}")
        for i, batch in enumerate(data_loader):
            print(f"Batch {i + 1}: {batch}")


        

   
