"""
This file (`data_utils.py`) provides utility functions and classes for data handling in deep learning models.
It includes tools for moving tensors to specific devices, load-balancing utilities for distributed training,
and custom samplers for PyTorch DataLoaders that support resumable training and balanced data distribution.

Key components:
- Recursive device transfer functionality
- Load balancing utilities for distributing data across processes
- Cyclical iteration through data loaders
- Custom resumable samplers for distributed training
"""

from typing import *
import math
import torch
import numpy as np
from torch.utils.data import Sampler, Dataset, DataLoader, DistributedSampler
import torch.distributed as dist


def recursive_to_device(
    data: Any,
    device: torch.device,
    non_blocking: bool = False,
) -> Any:
    """
    Recursively move all tensors in a data structure to a device.
    
    This function traverses nested data structures (lists, tuples, dictionaries)
    and moves any PyTorch tensor to the specified device.
    
    Args:
        data: The data structure containing tensors to be moved
        device: The target device (CPU, GPU) to move tensors to
        non_blocking: If True, allows asynchronous copy to device if possible
        
    Returns:
        The same data structure with all tensors moved to the specified device
    """
    if hasattr(data, "to"):
        # print("Moving data to device")
        # print(data)
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        # print("list or tuple detected") 
        return type(data)(recursive_to_device(d, device, non_blocking) for d in data)
    elif isinstance(data, dict):
        # print("dict detected")
        return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
    else:
        # print(f"{type(data)} detected")
        return data


def load_balanced_group_indices(
    load: List[int],
    num_groups: int,
    equal_size: bool = False,
) -> List[List[int]]:
    """
    Split indices into groups with balanced load.
    
    This function distributes indices across groups to achieve balanced workload.
    It uses a greedy algorithm that assigns each index to the group with the 
    minimum current load.
    
    Args:
        load: List of load values for each index
        num_groups: Number of groups to split indices into
        equal_size: If True, each group will have the same number of elements
        
    Returns:
        List of lists, where each inner list contains indices assigned to a group
    """
    if equal_size:
        group_size = len(load) // num_groups
    indices = np.argsort(load)[::-1]  # Sort indices by load in descending order
    groups = [[] for _ in range(num_groups)]
    group_load = np.zeros(num_groups)
    for idx in indices:
        min_group_idx = np.argmin(group_load)
        groups[min_group_idx].append(idx)
        if equal_size and len(groups[min_group_idx]) == group_size:
            group_load[min_group_idx] = float('inf')  # Mark group as full
        else:
            group_load[min_group_idx] += load[idx]
    return groups


def cycle(data_loader: DataLoader) -> Iterator:
    """
    Creates an infinite iterator over a data loader.
    
    This function wraps a data loader to cycle through it repeatedly, 
    handling epoch tracking for various sampler types.
    
    Args:
        data_loader: The DataLoader to cycle through
        
    Returns:
        An iterator that indefinitely yields batches from the data loader
    """
    while True:
        for data in data_loader:
            if isinstance(data_loader.sampler, ResumableSampler):
                data_loader.sampler.idx += data_loader.batch_size   # Update position for resumability
            yield data
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.epoch += 1  # Update epoch for DistributedSampler
        if isinstance(data_loader.sampler, ResumableSampler):
            data_loader.sampler.epoch += 1  # Update epoch for ResumableSampler
            data_loader.sampler.idx = 0     # Reset position index
        

class ResumableSampler(Sampler):
    """
    Distributed sampler that is resumable.

    This sampler extends PyTorch's Sampler to support resuming training from
    a specific point. It tracks the current position (idx) and epoch to 
    enable checkpointing and resuming.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.epoch = 0       # Current epoch counter
        self.idx = 0         # Current index position for resuming
        self.drop_last = drop_last
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1  # Get total number of processes
        self.rank = dist.get_rank() if dist.is_initialized() else 0              # Get current process rank
        
        # Calculate number of samples per process
        if self.drop_last and len(self.dataset) % self.world_size != 0:  
            # Split to nearest available length that is evenly divisible
            # This ensures each rank receives the same amount of data
            self.num_samples = math.ceil(
                (len(self.dataset) - self.world_size) / self.world_size  
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.world_size)  
            
        self.total_size = self.num_samples * self.world_size  # Total size after padding
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  
        else:
            indices = list(range(len(self.dataset)))  

        if not self.drop_last:
            # Add extra samples to make it evenly divisible across processes
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]  # Reuse some samples from the beginning
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]  # Repeat samples if padding_size > len(indices)
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # Subsample according to rank for distributed training
        indices = indices[self.rank : self.total_size : self.world_size]
        
        # Resume from previous state by skipping already processed indices
        indices = indices[self.idx:]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> Dict[str, int]:
        """
        Returns the state of the sampler as a dictionary.
        
        This enables saving the sampler state for checkpointing.
        
        Returns:
            Dictionary containing epoch and current index
        """
        return {
            'epoch': self.epoch,
            'idx': self.idx,
        }
        
    def load_state_dict(self, state_dict):
        """
        Loads the sampler state from a dictionary.
        
        This enables restoring the sampler state from a checkpoint.
        
        Args:
            state_dict: Dictionary containing sampler state
        """
        self.epoch = state_dict['epoch']
        self.idx = state_dict['idx']
        

class BalancedResumableSampler(ResumableSampler):
    """
    Distributed sampler that is resumable and balances the load among the processes.

    This sampler extends ResumableSampler to distribute data across processes
    in a load-balanced manner, ensuring that each process receives a similar
    computational workload despite potentially varying sample processing times.

    Args:
        dataset: Dataset used for sampling. Must have 'loads' attribute.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        batch_size (int, optional): Size of mini-batches used for balancing. Default: 1.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
    ) -> None:
        assert hasattr(dataset, 'loads'), 'Dataset must have "loads" attribute to use BalancedResumableSampler'
        super().__init__(dataset, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.loads = dataset.loads  # Load values for each sample in the dataset
        
    def __iter__(self) -> Iterator:
        # print(f"[BalancedResumableSampler] Starting __iter__ for rank {self.rank}, epoch {self.epoch}")
        
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # print(f"[BalancedResumableSampler] Shuffling with seed {self.seed + self.epoch}") # 0
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  
        else:
            # print(f"[BalancedResumableSampler] No shuffle, using sequential indices")
            indices = list(range(len(self.dataset)))  
        # print(indices)
        # print(f"[BalancedResumableSampler] Initial indices length: {len(indices)}") # 128
        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            # print(f"[BalancedResumableSampler] Adding padding of size {padding_size}") # 0
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            # print(f"[BalancedResumableSampler] Dropping last, trimming to {self.total_size}") 
            indices = indices[: self.total_size]
        # print(indices)
        assert len(indices) == self.total_size
        # print(f"[BalancedResumableSampler] After padding/trimming, indices length: {len(indices)}") # 128

        # Balance load among processes by distributing batches based on their loads
        num_batches = len(indices) // (self.batch_size * self.world_size)
        # print(f"[BalancedResumableSampler] Number of batches: {num_batches}") # 16
        balanced_indices = []

        if len(self.loads) < len(indices):
            # repeat the loads to match the indices
            self.loads = self.loads * (len(indices) // len(self.loads)) + self.loads[:len(indices) % len(self.loads)]

        for i in range(num_batches):
            start_idx = i * self.batch_size * self.world_size
            end_idx = (i + 1) * self.batch_size * self.world_size
            # print("start idx", start_idx) # 0
            # print("end idx", end_idx) # 8
            # print("batch size", self.batch_size) # 8
            # print("world size", self.world_size) # 1
            batch_indices = indices[start_idx:end_idx]
            # print(f"[BalancedResumableSampler] Processing batch {i+1}/{num_batches}, size: {len(batch_indices)}") #1/16 8
            batch_loads = [self.loads[idx] for idx in batch_indices]
            groups = load_balanced_group_indices(batch_loads, self.world_size, equal_size=True)
            balanced_indices.extend([batch_indices[j] for j in groups[self.rank]])
        
        # print(f"[BalancedResumableSampler] Total balanced indices for rank {self.rank}: {len(balanced_indices)}")
        # Resume from previous state
        indices = balanced_indices[self.idx:]
        # print(f"[BalancedResumableSampler] After resuming from idx {self.idx}, returning {len(indices)} indices")
        return iter(indices)


class DuplicatedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that duplicates a dataset multiple times."""
    
    def __init__(self, dataset, repeat=1000):
        """
        Initialize the duplicated dataset.
        
        Args:
            dataset: Original dataset to duplicate
            repeat: Number of times to repeat the dataset
        """
        self.dataset = dataset
        self.repeat = repeat
        self.original_length = len(dataset)
        
    def __getitem__(self, idx):
        """Get an item from the original dataset, repeating as needed."""
        return self.dataset[idx % self.original_length]
    
    def __len__(self):
        """Return the length of the duplicated dataset."""
        return self.original_length * self.repeat
    
    def __getattr__(self, name):
        """Forward all other attribute accesses to the original dataset."""
        if name == 'dataset' or name == 'repeat' or name == 'original_length':
            return object.__getattribute__(self, name)
        return getattr(self.dataset, name)

def save_coords_as_ply(coords, save_dir: str):
    """
    Save the coordinates to a PLY file using normalization similar to voxelize.py.

    Args:
        file_path (str): The directory path to save the PLY file.
    """
    import os
    # import numpy as np

    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Get coordinates and convert to numpy
    coords_np = coords.cpu().numpy()
    
    # Print debug info
    # print(f"Original coordinates shape: {coords_np.shape}")
    # print(f"First few coordinates:\n{coords_np[:5]}")
    
    if coords_np.shape[1] == 4:
    # Extract XYZ coordinates (skip batch index at position 0)
        vertices = coords_np[:, 1:4]
    else:
        vertices = coords_np

    # Normalize coordinates to [-0.5, 0.5] like in voxelize.py
    # Assuming the coordinates are in a 64³ grid
    GRID_SIZE = 64
    vertices = (vertices + 0.5) / GRID_SIZE - 0.5
    
    # print(f"Normalized vertex range: min={np.min(vertices, axis=0)}, max={np.max(vertices, axis=0)}")
    
    # Create PLY file (simplified format like in voxelize.py)
    filename = os.path.join(save_dir, 'coords.ply')
    
    try:
        with open(filename, 'w') as f:
            # Write header (no color, just XYZ coordinates)
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {vertices.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # Write vertices (no color)
            for i in range(vertices.shape[0]):
                f.write(f"{vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}\n")
        
        # print(f"PLY file saved to {filename} with {vertices.shape[0]} points")
        
        # Verify file creation
        # file_size = os.path.getsize(filename)
        # print(f"File size: {file_size} bytes")
            
    except Exception as e:
        print(f"Error creating PLY file: {e}")
    
    return filename