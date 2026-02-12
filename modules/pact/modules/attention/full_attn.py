"""
Full Attention Module

This file implements different versions of the Scaled Dot-Product Attention mechanism used in transformer models.
It provides a unified interface that supports multiple backend implementations (xformers, flash_attn, 
PyTorch's native SDPA, or a naive implementation) while maintaining consistent input/output formats.
The module allows for flexible calling patterns with different tensor arrangements for queries, keys, and values.
"""

from typing import *
import torch
import math
from . import DEBUG, BACKEND  # Import configuration variables

# Select the appropriate attention backend based on configuration
if BACKEND == 'xformers':
    import xformers.ops as xops
elif BACKEND == 'flash_attn':
    import flash_attn
elif BACKEND == 'sdpa':
    from torch.nn.functional import scaled_dot_product_attention as sdpa
elif BACKEND == 'naive':
    pass  # Will use the naive implementation defined below
else:
    raise ValueError(f"Unknown attention backend: {BACKEND}")


__all__ = [
    'scaled_dot_product_attention',  # Only expose this main function
]


def _naive_sdpa(q, k, v):
    """
    Naive implementation of scaled dot product attention.
    
    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        
    Returns:
        torch.Tensor: Output attention tensor
        
    Note:
        This implementation follows the standard attention formula:
        Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    """
    q = q.permute(0, 2, 1, 3)   # [N, H, L, C] - Reshape for batched matrix multiplication
    k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
    scale_factor = 1 / math.sqrt(q.size(-1))  # Scale factor to prevent softmax saturation
    attn_weight = q @ k.transpose(-2, -1) * scale_factor  # Compute scaled dot product
    attn_weight = torch.softmax(attn_weight, dim=-1)  # Apply softmax to get attention weights
    out = attn_weight @ v  # Apply attention weights to values
    out = out.permute(0, 2, 1, 3)   # [N, L, H, C] - Restore original dimension order
    return out


@overload
def scaled_dot_product_attention(qkv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        qkv (torch.Tensor): A [N, L, 3, H, C] tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, C] tensor containing Qs.
        kv (torch.Tensor): A [N, L, 2, H, C] tensor containing Ks and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

def scaled_dot_product_attention(*args, **kwargs):
    """
    Unified interface for scaled dot product attention with multiple calling patterns.
    
    Supports three calling patterns:
    1. Single combined QKV tensor: scaled_dot_product_attention(qkv)
    2. Separate Q and combined KV: scaled_dot_product_attention(q, kv)
    3. Separate Q, K, V tensors: scaled_dot_product_attention(q, k, v)
    
    The function automatically selects the appropriate backend implementation
    based on the BACKEND configuration.
    """
    # Define expected argument names for each calling pattern
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    # Handle case 1: Single combined QKV tensor
    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        device = qkv.device

    # Handle case 2: Separate Q and combined KV tensors
    elif num_all_args == 2:
        # print("handle case 2")
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        device = q.device

    # Handle case 3: Separate Q, K, V tensors
    elif num_all_args == 3:
        # print("handle case 3")
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
        device = q.device    

    # print("no problem")
    # Use xformers backend
    if BACKEND == 'xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)  # Split combined tensor into separate Q, K, V
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)  # Split combined KV tensor
        out = xops.memory_efficient_attention(q, k, v)
    
    # Use Flash Attention backend
    elif BACKEND == 'flash_attn':
        # print("flash_attn")
        if num_all_args == 1:
            # print("case 1")
            out = flash_attn.flash_attn_qkvpacked_func(qkv)  # Use packed QKV format
        elif num_all_args == 2:
            # print("case 2")
            out = flash_attn.flash_attn_kvpacked_func(q, kv)  # Use packed KV format with separate Q
        elif num_all_args == 3:
            # print("case 3")
            out = flash_attn.flash_attn_func(q, k, v)  # Use fully separate Q, K, V
    
    # Use PyTorch's native scaled dot product attention
    elif BACKEND == 'sdpa':
        # print("sdpa")
        if num_all_args == 1:
            # print("case 1")
            q, k, v = qkv.unbind(dim=2)  # Split combined tensor
        elif num_all_args == 2:
            # print("case 2")
            k, v = kv.unbind(dim=2)  # Split combined KV tensor
        # PyTorch's SDPA expects tensors in format [N, H, L, C]
        q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
        k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
        v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
        out = sdpa(q, k, v)         # [N, H, L, C]
        out = out.permute(0, 2, 1, 3)   # Convert back to [N, L, H, C]
    
    # Use naive implementation
    elif BACKEND == 'naive':
        # print("naive")
        if num_all_args == 1:
            # print("case 1")
            q, k, v = qkv.unbind(dim=2)  # Split combined tensor
        elif num_all_args == 2:
            # print("case 2")
            k, v = kv.unbind(dim=2)  # Split combined KV tensor
        out = _naive_sdpa(q, k, v)  # Call the naive implementation
    
    else:
        raise ValueError(f"Unknown attention module: {BACKEND}")
    # print("no problem")
    return out
