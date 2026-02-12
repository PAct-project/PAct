"""
Base Sparse Transformer Implementation for TRELLIS Framework

This file implements the base architecture for sparse transformers used in structured latent variable models.
It provides a configurable foundation with multiple attention mechanisms (full, windowed, shifted window)
and supports different positional encoding strategies. The sparse implementation allows for efficient
processing of data with varying density patterns.

The main class SparseTransformerBase serves as the foundation for encoder and decoder implementations
in the structured latent VAE models.
"""

from typing import *
import torch
import torch.nn as nn
from ...modules.utils import convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from ...modules.transformer import AbsolutePositionEmbedder
from ...modules.sparse.transformer import SparseTransformerBlock


def block_attn_config(self):
    """
    Return the attention configuration for each transformer block.
    
    Generates configurations for each block based on the specified attention mode:
    - shift_window: Uses serialized attention with shifting window patterns
    - shift_sequence: Uses serialized attention with sequence shifts
    - shift_order: Uses serialized attention with different serialization orders
    - full: Uses standard full attention (non-sparse)
    - swin: Uses Swin Transformer-style windowed attention
    
    Yields:
        Tuple containing attention mode and its parameters
    """
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class SparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    
    Implements a transformer architecture that can work with sparse data structures,
    supporting various attention mechanisms and positional encodings.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        """
        Initialize the sparse transformer base model.
        
        Args:
            in_channels: Number of input channels
            model_channels: Hidden dimension size
            num_blocks: Number of transformer blocks
            num_heads: Number of attention heads (calculated from head_channels if None)
            num_head_channels: Number of channels per attention head
            mlp_ratio: Ratio for MLP hidden dimension
            attn_mode: Attention mechanism type
            window_size: Size of attention window for windowed modes
            pe_mode: Positional encoding mode (absolute or rotary)
            use_fp16: Whether to use half precision
            use_checkpoint: Whether to use gradient checkpointing
            qk_rms_norm: Whether to use RMS normalization for query and key
        """
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # Create positional embedder if using absolute positional encoding
        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        # Input projection layer
        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        
        # Build transformer blocks with configurations from block_attn_config
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16 precision.
        Used for mixed precision training.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model back to float32 precision.
        Used after mixed precision training or inference.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the model using Xavier uniform initialization.
        This helps with training stability and convergence.
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Forward pass through the sparse transformer.
        
        Args:
            x: Input sparse tensor
            
        Returns:
            Processed sparse tensor after passing through all transformer blocks
        """
        # Project input to model dimension
        h = self.input_layer(x)
        
        # Add positional embeddings if using absolute positional encoding
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        
        # Convert to target precision
        h = h.type(self.dtype)
        
        # Pass through transformer blocks sequentially
        for block in self.blocks:
            h = block(h)
            
        return h
