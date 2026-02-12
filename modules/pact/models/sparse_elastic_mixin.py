"""
This file defines a mixin class for sparse transformers that enables elastic memory management.
It provides functionality to dynamically adjust memory usage by controlling gradient checkpointing
across transformer blocks, allowing for trading computation for memory efficiency.
"""

from contextlib import contextmanager
from typing import *
import math
from ..modules import sparse as sp
from ..utils.elastic_utils import ElasticModuleMixin


class SparseTransformerElasticMixin(ElasticModuleMixin):
    """
    A mixin class for sparse transformers that provides elastic memory management capabilities.
    Extends the base ElasticModuleMixin with sparse tensor-specific functionality.
    """
    
    def _get_input_size(self, x: sp.SparseTensor, *args, **kwargs):
        """
        Determines the input size from a sparse tensor.
        
        Args:
            x: A SparseTensor input
            *args, **kwargs: Additional arguments (unused)
            
        Returns:
            The size of the feature dimension of the sparse tensor
        """
        return x.feats.shape[0]
    
    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0):
        """
        Context manager that temporarily adjusts memory usage by enabling gradient checkpointing
        for a portion of the transformer blocks based on the specified memory ratio.
        
        Args:
            mem_ratio: A value between 0 and 1 indicating the desired memory ratio.
                      1.0 means use all available memory (no checkpointing).
                      Lower values enable more checkpointing to reduce memory usage.
        
        Yields:
            The exact memory ratio that could be achieved with the block granularity.
        """
        if mem_ratio == 1.0:
            # No memory optimization needed if ratio is 1.0
            yield 1.0
            return
            
        # Calculate how many blocks should use checkpointing
        num_blocks = len(self.blocks)
        num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
        
        # Calculate the actual memory ratio based on the number of checkpointed blocks
        exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
        
        # Enable checkpointing for the calculated number of blocks
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = i < num_checkpoint_blocks
            
        yield exact_mem_ratio
        
        # Restore all blocks to not use checkpointing after context exit
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = False
