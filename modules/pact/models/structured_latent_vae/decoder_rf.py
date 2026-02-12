"""
This file implements radiance field decoders for Structured Latent VAE models.
The main class SLatRadianceFieldDecoder is a sparse transformer-based decoder that 
transforms latent codes into sparse representations of 3D scenes (Strivec representation).
It also includes an elastic memory version (ElasticSLatRadianceFieldDecoder) for low VRAM training.
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ...representations import Strivec
from ..sparse_elastic_mixin import SparseTransformerElasticMixin


class SLatRadianceFieldDecoder(SparseTransformerBase):
    """
    A sparse transformer-based decoder for converting latent codes to radiance field representations.
    This decoder processes sparse tensors through transformer blocks and outputs parameters for Strivec representation.
    """
    def __init__(
        self,
        resolution: int,  # Resolution of the output 3D grid
        model_channels: int,  # Number of channels in the model's hidden layers
        latent_channels: int,  # Number of channels in the latent code
        num_blocks: int,  # Number of transformer blocks
        num_heads: Optional[int] = None,  # Number of attention heads
        num_head_channels: Optional[int] = 64,  # Channels per attention head
        mlp_ratio: float = 4,  # Ratio for MLP hidden dimension
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",  # Attention mode
        window_size: int = 8,  # Size of local attention window
        pe_mode: Literal["ape", "rope"] = "ape",  # Positional encoding mode
        use_fp16: bool = False,  # Whether to use half precision
        use_checkpoint: bool = False,  # Whether to use gradient checkpointing
        qk_rms_norm: bool = False,  # Whether to normalize query and key
        representation_config: dict = None,  # Configuration for output representation
    ):
        # Initialize the base sparse transformer
        super().__init__(
            in_channels=latent_channels,
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
        self.rep_config = representation_config
        self._calc_layout()  # Calculate the output layout
        # Final layer to project features to the output representation
        self.out_layer = sp.SparseLinear(model_channels, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the model.
        Zero-initializes the output layer for better training stability.
        """
        super().initialize_weights()
        # Zero-out output layers for better training stability
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _calc_layout(self) -> None:
        """
        Calculate the output tensor layout for the Strivec representation.
        Defines the shapes and sizes of different components and their positions in the output tensor.
        """
        self.layout = {
            'trivec': {'shape': (self.rep_config['rank'], 3, self.rep_config['dim']), 'size': self.rep_config['rank'] * 3 * self.rep_config['dim']},
            'density': {'shape': (self.rep_config['rank'],), 'size': self.rep_config['rank']},
            'features_dc': {'shape': (self.rep_config['rank'], 1, 3), 'size': self.rep_config['rank'] * 3},
        }
        # Calculate the range (start, end) indices for each component in the output tensor
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.out_channels = start    
    
    def to_representation(self, x: sp.SparseTensor) -> List[Strivec]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of Strivec representations, one per batch item
        """
        ret = []
        for i in range(x.shape[0]):
            # Create a new Strivec representation
            representation = Strivec(
                sh_degree=0,
                resolution=self.resolution,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],  # Axis-aligned bounding box
                rank=self.rep_config['rank'],
                dim=self.rep_config['dim'],
                device='cuda',
            )
            representation.density_shift = 0.0
            # Set position from sparse coordinates (normalized to [0,1])
            representation.position = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            # Set depth (octree level) based on resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(self.resolution)), dtype=torch.uint8, device='cuda')
            
            # Extract each component from the output features according to the layout
            for k, v in self.layout.items():
                setattr(representation, k, x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']))
            
            # Add 1 to trivec for stability (prevent zero vectors)
            representation.trivec = representation.trivec + 1
            ret.append(representation)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[Strivec]:
        """
        Forward pass through the decoder.
        
        Args:
            x: Input sparse tensor containing latent codes
            
        Returns:
            List of Strivec representations
        """
        # Pass through transformer backbone
        h = super().forward(x)
        h = h.type(x.dtype)
        # Layer normalization on feature dimension
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        # Final projection to output features
        h = self.out_layer(h)
        # Convert network output to Strivec representations
        return self.to_representation(h)


class ElasticSLatRadianceFieldDecoder(SparseTransformerElasticMixin, SLatRadianceFieldDecoder):
    """
    Slat VAE Radiance Field Decoder with elastic memory management.
    Used for training with low VRAM by dynamically managing memory allocation
    and performing operations in chunks when needed.
    """
    pass
