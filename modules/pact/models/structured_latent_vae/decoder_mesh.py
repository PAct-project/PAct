"""
Mesh Decoder Module for Structured Latent VAE

This file implements a mesh-based decoder for the structured latent variational autoencoder (SLAT VAE).
It contains specialized sparse neural network components that transform latent representations into 
3D mesh structures through a series of sparse convolutions and subdivisions.

The module implements:
1. SparseSubdivideBlock3d - A block that subdivides sparse tensors to increase resolution
2. SLatMeshDecoder - Main decoder that transforms latent codes into 3D meshes
3. ElasticSLatMeshDecoder - Memory-efficient version for low VRAM environments
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ...representations import MeshExtractResult
from ...representations.mesh import SparseFeatures2Mesh
from ..sparse_elastic_mixin import SparseTransformerElasticMixin


class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    This block increases the resolution of sparse tensors by a factor of 2,
    and optionally changes the number of channels.

    Args:
        channels: channels in the inputs and outputs.
        resolution: the current resolution of the sparse tensor.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        # Normalization and activation before subdivision
        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )
        
        # Subdivision operator that doubles the resolution
        self.sub = sp.SparseSubdivide()
        
        # Post-subdivision processing with residual connection
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        # Skip connection that handles potential channel dimension changes
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs with doubled resolution.
        """
        h = self.act_layers(x)
        h = self.sub(h)  # Double the resolution
        x = self.sub(x)  # Also subdivide the input for skip connection
        h = self.out_layers(h)
        h = h + self.skip_connection(x)  # Add skip connection
        return h


class SLatMeshDecoder(SparseTransformerBase):
    """
    Structured Latent Mesh Decoder that transforms latent codes into 3D meshes.
    
    Uses sparse transformers followed by upsampling blocks to generate high-resolution
    features that are then converted to meshes.
    """
    def __init__(
        self,
        resolution: int,
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
        representation_config: dict = None,
    ):
        # Initialize the transformer backbone
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
        
        # Mesh extractor to convert features to mesh representation
        self.mesh_extractor = SparseFeatures2Mesh(res=self.resolution*4, use_color=self.rep_config.get('use_color', False))
        self.out_channels = self.mesh_extractor.feats_channels
        
        # Upsampling blocks that progressively increase resolution
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=resolution,
                out_channels=model_channels // 4
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8
            )
        ])
        
        # Final layer to map features to mesh attributes
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        """Initialize model weights, with special handling for output layers."""
        super().initialize_weights()
        # Zero-out output layers for stable training
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16 for memory efficiency.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model back to float32 for precision.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)  
    
    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to 3D mesh representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of mesh representation results, one per batch item
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Process latent codes through the decoder and extract meshes.
        
        Args:
            x: Input sparse tensor of latent codes
            
        Returns:
            List of extracted mesh representations
        """
        h = super().forward(x)  # Process through transformer blocks
        for block in self.upsample:
            h = block(h)  # Progressively increase resolution
        h = h.type(x.dtype)
        h = self.out_layer(h)  # Final projection to mesh features
        return self.to_representation(h)  # Convert features to meshes
    

class ElasticSLatMeshDecoder(SparseTransformerElasticMixin, SLatMeshDecoder):
    """
    Structured Latent Mesh Decoder with elastic memory management.
    
    This variant uses elastic sparse tensor operations to reduce memory usage
    during training, making it suitable for environments with limited VRAM.
    """
    pass
