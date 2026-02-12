"""
sparse_structure_vae.py

This file implements a Variational Autoencoder (VAE) for 3D sparse structural representations.
It's part of the TRELLIS framework and contains components for encoding volumetric data
into a latent space and decoding it back to volumetric representation.

The implementation includes:
- 3D normalization layers
- 3D residual blocks for feature extraction
- 3D downsampling and upsampling blocks for resolution changes
- Encoder (SparseStructureEncoder) that maps input volumes to a latent distribution
- Decoder (SparseStructureDecoder) that reconstructs volumes from latent codes

This VAE architecture is specifically designed for capturing structural information
in a compressed latent representation that can be sampled probabilistically.
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.norm import GroupNorm32, ChannelLayerNorm32
from ..modules.spatial import pixel_shuffle_3d
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer based on the specified type.
    
    Args:
        norm_type: Either "group" for GroupNorm or "layer" for LayerNorm
        *args, **kwargs: Arguments passed to the normalization layer
        
    Returns:
        An instance of the requested normalization layer
    """
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    """
    3D Residual Block with two convolutions and a skip connection.
    
    The block applies normalization, activation, and convolution twice,
    with a skip connection from the input to the output.
    """
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        """
        Initialize a 3D ResBlock.
        
        Args:
            channels: Number of input channels
            out_channels: Number of output channels (defaults to input channels)
            norm_type: Type of normalization to use
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        # First normalization and convolution
        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        # Second convolution is initialized with zeros for stable training
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        # Skip connection: identity if channels match, otherwise 1x1 conv
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResBlock.
        
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            
        Returns:
            Output tensor after residual computation
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)  # Residual connection
        return h


class DownsampleBlock3d(nn.Module):
    """
    3D downsampling block to reduce spatial dimensions by a factor of 2.
    
    Supports downsampling via strided convolution or average pooling.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        """
        Initialize a 3D downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mode: Downsampling method ("conv" or "avgpool")
        """
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the downsampling block.
        
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            
        Returns:
            Downsampled tensor
        """
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    """
    3D upsampling block to increase spatial dimensions by a factor of 2.
    
    Supports upsampling via transposed convolution or nearest-neighbor interpolation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        """
        Initialize a 3D upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mode: Upsampling method ("conv" or "nearest")
        """
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            # For pixel shuffle upsampling, we need 8x channels (2³ = 8)
            self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)
        elif mode == "nearest":
            assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the upsampling block.
        
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            
        Returns:
            Upsampled tensor
        """
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)  # 3D pixel shuffle for upsampling
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        

class SparseStructureEncoder(nn.Module):
    """
    Encoder for Sparse Structure (\mathcal{E}_S in the paper Sec. 3.3).
    
    Takes a 3D volume as input and encodes it into a latent distribution (mean and logvar).
    Can sample from this distribution to get a latent representation.
    
    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        """
        Initialize the encoder for sparse structure.
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # Initial projection from input to feature space
        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        # Encoder blocks with progressive downsampling
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            # Add residual blocks at the current resolution
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            # Add downsampling block if not at the final resolution
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i+1])
                )
        
        # Middle blocks at the lowest resolution
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

        # Output layer produces both mean and logvar for the latent distribution
        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels*2, 3, padding=1)
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            sample_posterior: Whether to sample from the posterior distribution or just return mean
            return_raw: Whether to return the raw outputs (z, mean, logvar) instead of just z
            
        Returns:
            Either the latent representation or a tuple of (z, mean, logvar) if return_raw=True
        """
        h = self.input_layer(x)
        h = h.type(self.dtype)  # Convert to FP16 if needed

        # Process through encoder blocks
        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)

        h = h.type(x.dtype)  # Convert back to input dtype
        h = self.out_layer(h)

        # Split output into mean and log variance
        mean, logvar = h.chunk(2, dim=1)

        # Sample from the posterior if requested
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)  # Reparameterization trick
        else:
            z = mean
            
        if return_raw:
            return z, mean, logvar
        return z
        

class SparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure (\mathcal{D}_S in the paper Sec. 3.3).
    
    Takes a latent representation and decodes it back to a 3D volume.
    Uses a symmetric architecture to the encoder with upsampling instead of downsampling.
    
    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """ 
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        """
        Initialize the decoder for sparse structure.
        """
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # Initial projection from latent space to feature space
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        # Middle blocks at the lowest resolution
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])

        # Decoder blocks with progressive upsampling
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            # Add residual blocks at the current resolution
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            # Add upsampling block if not at the final resolution
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i+1])
                )

        # Final output layer to generate the desired output channels
        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1)
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x: Latent representation tensor of shape [B, C, D, H, W]
            
        Returns:
            Reconstructed output tensor
        """
        h = self.input_layer(x)
        
        h = h.type(self.dtype)  # Convert to FP16 if needed
                
        h = self.middle_block(h)
        # Process through decoder blocks
        for block in self.blocks:
            h = block(h)

        h = h.type(x.dtype)  # Convert back to input dtype
        h = self.out_layer(h)
        return h
