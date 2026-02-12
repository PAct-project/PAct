"""
Structured Latent Variable Encoder Module
----------------------------------------
This file defines encoder classes for the Structured Latent Variable Autoencoder (SLatVAE).
It contains implementations for the sparse transformer-based encoder that maps input 
features to a latent distribution, as well as a memory-efficient elastic version.
The encoder follows a variational approach, outputting means and log variances for
the latent space representation.
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ..sparse_elastic_mixin import SparseTransformerElasticMixin


class SLatEncoder(SparseTransformerBase):
    """
    Sparse Latent Variable Encoder that uses transformer architecture to encode
    sparse data into a latent distribution.
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        """
        Initialize the Sparse Latent Encoder.
        
        Args:
            resolution: Input data resolution
            in_channels: Number of input feature channels
            model_channels: Number of internal model feature channels
            latent_channels: Dimension of the latent space
            num_blocks: Number of transformer blocks
            num_heads: Number of attention heads (optional)
            num_head_channels: Channels per attention head if num_heads is None
            mlp_ratio: Expansion ratio for MLP in transformer blocks
            attn_mode: Type of attention mechanism to use
            window_size: Size of attention windows if using windowed attention
            pe_mode: Positional encoding mode (absolute or relative)
            use_fp16: Whether to use half-precision floating point
            use_checkpoint: Whether to use gradient checkpointing
            qk_rms_norm: Whether to apply RMS normalization to query and key
        """
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        # Output layer projects to twice the latent dimension (for mean and logvar)
        self.out_layer = sp.SparseLinear(model_channels, 2 * latent_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        """
        Initialize model weights with special handling for output layer.
        The output layer weights are initialized to zero to stabilize training.
        """
        super().initialize_weights()
        # Zero-out output layers for better training stability
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, sample_posterior=True, return_raw=False):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input sparse tensor
            sample_posterior: Whether to sample from posterior or return mean
            return_raw: Whether to return mean and logvar in addition to samples
            
        Returns:
            If return_raw is True:
                - sampled latent variables, mean, and logvar
            Otherwise:
                - sampled latent variables only
        """
        # Process through transformer blocks
        h = super().forward(x)
        h = h.type(x.dtype)
        # Apply layer normalization to features
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        
        # Split output into mean and logvar components
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            # Reparameterization trick: z = mean + std * epsilon
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            # Use mean directly without sampling
            z = mean
        z = h.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z
        

class ElasticSLatEncoder(SparseTransformerElasticMixin, SLatEncoder):
    """
    SLat VAE encoder with elastic memory management.
    Used for training with low VRAM by dynamically managing memory allocation
    and performing operations with reduced memory footprint.
    """
    pass
