"""
Distributed Training Utilities

This file contains utility functions for distributed training with PyTorch.
It provides tools for setting up distributed environments, efficient file handling
across processes, model unwrapping, and synchronization mechanisms to coordinate 
execution across multiple GPUs and nodes.
"""

import os
import io
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_dist(rank, local_rank, world_size, master_addr, master_port):
    """
    Set up the distributed training environment.
    
    Args:
        rank (int): Global rank of the current process
        local_rank (int): Local rank of the current process on this node
        world_size (int): Total number of processes in the distributed training
        master_addr (str): IP address of the master node
        master_port (str): Port on the master node for communication
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    # Set the device for the current process
    torch.cuda.set_device(local_rank)
    # Initialize the process group for distributed communication
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    

def read_file_dist(path):
    """
    Read the binary file distributedly.
    File is only read once by the rank 0 process and broadcasted to other processes.
    This reduces I/O overhead in distributed training.

    Args:
        path (str): Path to the file to be read

    Returns:
        data (io.BytesIO): The binary data read from the file.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        # Prepare tensor to store file size
        size = torch.LongTensor(1).cuda()
        if dist.get_rank() == 0:
            # Master process reads the file
            with open(path, 'rb') as f:
                data = f.read()
            # Convert binary data to CUDA tensor for broadcasting
            data = torch.ByteTensor(
                torch.UntypedStorage.from_buffer(data, dtype=torch.uint8)
            ).cuda()
            size[0] = data.shape[0]
        # Broadcast file size to all processes
        dist.broadcast(size, src=0)
        if dist.get_rank() != 0:
            # Non-master processes allocate buffer for receiving data
            data = torch.ByteTensor(size[0].item()).cuda()
        # Broadcast actual file data to all processes
        dist.broadcast(data, src=0)
        # Convert tensor back to binary data
        data = data.cpu().numpy().tobytes()
        data = io.BytesIO(data)
        return data
    else:
        # For non-distributed or single-process case, just read directly
        with open(path, 'rb') as f:
            data = f.read()
        data = io.BytesIO(data)
        return data
    

def unwrap_dist(model):
    """
    Unwrap the model from distributed training wrapper.
    
    Args:
        model: A potentially wrapped PyTorch model
        
    Returns:
        The underlying model without DistributedDataParallel wrapper
    """
    if isinstance(model, DDP):
        return model.module
    return model


@contextmanager
def master_first():
    """
    A context manager that ensures master process (rank 0) executes first.
    All other processes wait for the master to finish before proceeding.
    
    Usage:
        with master_first():
            # Code that should execute in master first, then others
    """
    if not dist.is_initialized():
        # If not in distributed mode, just execute normally
        yield
    else:
        if dist.get_rank() == 0:
            # Master process executes the code
            yield
            # Signal completion to other processes
            dist.barrier()
        else:
            # Other processes wait for master to finish
            dist.barrier()
            # Then execute the code
            yield
            

@contextmanager
def local_master_first():
    """
    A context manager that ensures local master process (first process on each node)
    executes first. Other processes on the same node wait before proceeding.
    
    Usage:
        with local_master_first():
            # Code that should execute in local master first, then others
    """
    if not dist.is_initialized():
        # If not in distributed mode, just execute normally
        yield
    else:
        if dist.get_rank() % torch.cuda.device_count() == 0:
            # Local master process executes the code
            yield
            # Signal completion to other processes
            dist.barrier()
        else:
            # Other processes wait for local master to finish
            dist.barrier()
            # Then execute the code
            yield