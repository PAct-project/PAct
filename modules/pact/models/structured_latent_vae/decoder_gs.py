"""
decoder_gs.py: Structured Latent Gaussian Decoder for 3D Representation Learning

This file contains decoder implementations that transform latent codes into 3D Gaussian
representations. The decoders use sparse transformer architectures for efficient processing
and flexible attention mechanisms. The main components are:
- SLatGaussianDecoder: Core decoder that maps latent codes to 3D Gaussians
- ElasticSLatGaussianDecoder: Memory-efficient variant with elastic memory management
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from ...utils.random_utils import hammersley_sequence
from .base import SparseTransformerBase
from ...representations import Gaussian
from ..sparse_elastic_mixin import SparseTransformerElasticMixin


class SLatGaussianDecoder(SparseTransformerBase):
    """
    Sparse Transformer-based decoder that converts latent codes to 3D Gaussian representations.
    
    This decoder processes sparse tensors and outputs parameters for Gaussian primitives
    that can be rendered in 3D space, including positions, features, scaling, rotation,
    and opacity.
    """
    def __init__(
        self,
        resolution: int,  # The resolution of the 3D grid
        model_channels: int,  # Number of channels in the transformer layers
        latent_channels: int,  # Number of channels in the input latent code
        num_blocks: int,  # Number of transformer blocks
        num_heads: Optional[int] = None,  # Number of attention heads
        num_head_channels: Optional[int] = 64,  # Channels per attention head
        mlp_ratio: float = 4,  # Ratio for MLP size in transformer blocks
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",  # Attention mechanism
        window_size: int = 8,  # Size of attention windows for windowed attention
        pe_mode: Literal["ape", "rope"] = "ape",  # Positional encoding mode
        use_fp16: bool = False,  # Whether to use half-precision
        use_checkpoint: bool = False,  # Whether to use gradient checkpointing
        qk_rms_norm: bool = False,  # Whether to use RMS normalization for attention
        representation_config: dict = None,  # Configuration for the Gaussian representation
    ):
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
        self._calc_layout()  # Calculate output tensor layout
        self.out_layer = sp.SparseLinear(model_channels, self.out_channels)  # Final projection layer
        self._build_perturbation()  # Build position perturbation for better initialization

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        """
        Initialize model weights, with special handling for output layers.
        Zero-initializes the output layer for stability.
        """
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _build_perturbation(self) -> None:
        """
        Build position perturbation for Gaussian means.
        Uses Hammersley sequence for quasi-random uniform distribution of points,
        then transforms to match the desired Gaussian spatial distribution.
        """
        perturbation = [hammersley_sequence(3, i, self.rep_config['num_gaussians']) for i in range(self.rep_config['num_gaussians'])]
        perturbation = torch.tensor(perturbation).float() * 2 - 1  # Scale to [-1, 1]
        perturbation = perturbation / self.rep_config['voxel_size']  # Scale by voxel size
        perturbation = torch.atanh(perturbation).to(self.device)  # Apply inverse tanh for better gradient flow
        self.register_buffer('offset_perturbation', perturbation)  # Register as buffer (not a parameter)

    def _calc_layout(self) -> None:
        """
        Calculate the layout of the output tensor.
        Defines the shape and size of each Gaussian parameter group (position, features, scaling, rotation, opacity)
        and their positions in the output tensor.
        """
        self.layout = {
            '_xyz' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_features_dc' : {'shape': (self.rep_config['num_gaussians'], 1, 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_scaling' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_rotation' : {'shape': (self.rep_config['num_gaussians'], 4), 'size': self.rep_config['num_gaussians'] * 4},
            '_opacity' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
        }
        # Calculate ranges for each parameter group in the flattened output tensor
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.out_channels = start  # Total number of output channels
    
    def to_representation(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D Gaussian representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of Gaussian representations, one per batch item
        """
        ret = []
        for i in range(x.shape[0]):
            # Create a new Gaussian representation object with proper configuration
            representation = Gaussian(
                sh_degree=0,  # No spherical harmonics, just using DC term
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],  # Axis-aligned bounding box
                mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                scaling_bias = self.rep_config['scaling_bias'],
                opacity_bias = self.rep_config['opacity_bias'],
                scaling_activation = self.rep_config['scaling_activation']
            )
            # Get base positions from sparse tensor coordinates
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            
            # Process each parameter group
            for k, v in self.layout.items():
                if k == '_xyz':
                    # Handle positions with special perturbation logic
                    offset = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape'])
                    offset = offset * self.rep_config['lr'][k]  # Apply learning rate scale
                    if self.rep_config['perturb_offset']:
                        offset = offset + self.offset_perturbation  # Add perturbation
                    # Transform offsets through tanh and scale appropriately
                    offset = torch.tanh(offset) / self.resolution * 0.5 * self.rep_config['voxel_size']
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    # Handle other parameters (features, scaling, rotation, opacity)
                    feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                    feats = feats * self.rep_config['lr'][k]  # Apply parameter-specific learning rate
                    setattr(representation, k, feats)
            ret.append(representation)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Forward pass through the decoder.
        
        Args:
            x: Input sparse tensor containing latent codes
            
        Returns:
            List of Gaussian representations ready for rendering
        """
        h = super().forward(x)  # Process through transformer blocks
        h = h.type(x.dtype)  # Ensure consistent dtype
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))  # Apply layer normalization
        h = self.out_layer(h)  # Project to final output dimensions
        return self.to_representation(h)  # Convert to Gaussian representations
    

class ElasticSLatGaussianDecoder(SparseTransformerElasticMixin, SparseTransformerBase):
    """
    Slat VAE Gaussian decoder with elastic memory management.
    Used for training with low VRAM by dynamically managing memory allocations
    and using efficient sparse operations.
    """
    pass
