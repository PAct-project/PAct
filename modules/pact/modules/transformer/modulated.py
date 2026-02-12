"""
This file implements modulated transformer blocks for conditional generation.
These blocks extend standard transformer architectures by incorporating adaptive layer normalization (adaLN),
which modulates the transformer's behavior based on conditioning information.
The modulation is applied through shift and scale parameters derived from a condition vector,
allowing the model to adapt its processing to different inputs or conditions.

The file provides two main components:
1. ModulatedTransformerBlock: A standard transformer block with self-attention and FFN, modified with adaLN
2. ModulatedTransformerCrossBlock: An extended transformer block with self-attention, cross-attention, and FFN with adaLN
"""

from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention,PartBased_MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    
    This block combines multi-head self-attention with a feed-forward network,
    and uses adaptive layer normalization to condition the processing on external information.
    """
    def __init__(
        self,
        channels: int,           # Number of input/output channels
        num_heads: int,          # Number of attention heads
        mlp_ratio: float = 4.0,  # Ratio determining MLP hidden dimension size
        attn_mode: Literal["full", "windowed"] = "full",  # Attention computation mode
        window_size: Optional[int] = None,    # Size of attention window if using windowed attention
        shift_window: Optional[Tuple[int, int, int]] = None,  # Parameters for shifted window attention
        use_checkpoint: bool = False,  # Whether to use gradient checkpointing to save memory
        use_rope: bool = False,   # Whether to use Rotary Position Embedding
        qk_rms_norm: bool = False,  # Whether to use RMS normalization for query and key
        qkv_bias: bool = True,    # Whether to use bias in QKV projection
        share_mod: bool = False,  # Whether to share modulation parameters externally
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        
        # Layer normalization without affine parameters (will be modulated)
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        
        # Multi-head self-attention layer
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        
        # Feed-forward network
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        
        # Modulation network to generate adaptive parameters if not shared
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)  # 6 channels: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        """
        Internal forward function for the modulated transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, channels]
            mod: Modulation tensor [batch, channels]
            
        Returns:
            Processed tensor with same shape as input
        """
        # Split modulation vector into shift, scale, and gate parameters for MSA and FFN
        if self.share_mod:
            # Use externally provided modulation parameters
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            # Generate modulation parameters from the conditioning vector
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        
        # Apply modulated self-attention
        h = self.norm1(x)  # Normalize
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)  # Apply modulation
        h = self.attn(h)  # Self-attention
        h = h * gate_msa.unsqueeze(1)  # Apply gate
        x = x + h  # Residual connection
        
        # Apply modulated feed-forward network
        h = self.norm2(x)  # Normalize
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)  # Apply modulation
        h = self.mlp(h)  # Feed-forward
        h = h * gate_mlp.unsqueeze(1)  # Apply gate
        x = x + h  # Residual connection
        
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional gradient checkpointing to save memory.
        
        Args:
            x: Input tensor [batch, seq_len, channels]
            mod: Modulation tensor [batch, channels]
            
        Returns:
            Processed tensor with same shape as input
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    
    This block extends the standard transformer block with an additional cross-attention
    layer, allowing it to attend to a separate context input.
    """
    def __init__(
        self,
        channels: int,           # Number of input/output channels
        ctx_channels: int,       # Number of context channels
        num_heads: int,          # Number of attention heads
        mlp_ratio: float = 4.0,  # Ratio determining MLP hidden dimension size
        attn_mode: Literal["full", "windowed"] = "full",  # Attention computation mode
        window_size: Optional[int] = None,    # Size of attention window if using windowed attention
        shift_window: Optional[Tuple[int, int, int]] = None,  # Parameters for shifted window attention
        use_checkpoint: bool = False,  # Whether to use gradient checkpointing to save memory
        use_rope: bool = False,   # Whether to use Rotary Position Embedding
        qk_rms_norm: bool = False,  # Whether to use RMS normalization for query and key in self-attention
        qk_rms_norm_cross: bool = False,  # Whether to use RMS normalization for query and key in cross-attention
        qkv_bias: bool = True,    # Whether to use bias in QKV projection
        share_mod: bool = False,  # Whether to share modulation parameters externally
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        
        # Layer normalizations
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)  # For self-attention, will be modulated
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)  # For cross-attention, standard normalization
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)  # For FFN, will be modulated
        
        # Self-attention layer
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        
        # Cross-attention layer
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",  # Cross-attention always uses full attention
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        
        # Feed-forward network
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        
        # Modulation network to generate adaptive parameters if not shared
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)  # 6 channels: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        """
        Internal forward function for the modulated transformer cross-attention block.
        
        Args:
            x: Input tensor [batch, seq_len, channels]
            mod: Modulation tensor [batch, channels]
            context: Context tensor for cross-attention [batch, context_len, ctx_channels]
            
        Returns:
            Processed tensor with same shape as input
        """
        # Split modulation vector into shift, scale, and gate parameters for MSA and FFN
        if self.share_mod:
            # Use externally provided modulation parameters
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            # Generate modulation parameters from the conditioning vector
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        
        # Apply modulated self-attention
        h = self.norm1(x)  # Normalize
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)  # Apply modulation
        h = self.self_attn(h)  # Self-attention
        h = h * gate_msa.unsqueeze(1)  # Apply gate
        x = x + h  # Residual connection
        
        # Apply cross-attention (not modulated)
        h = self.norm2(x)  # Normalize
        h = self.cross_attn(h, context)  # Cross-attention with context
        x = x + h  # Residual connection
        
        # Apply modulated feed-forward network
        h = self.norm3(x)  # Normalize
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)  # Apply modulation
        h = self.mlp(h)  # Feed-forward
        h = h * gate_mlp.unsqueeze(1)  # Apply gate
        x = x + h  # Residual connection
        
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        """
        Forward pass with optional gradient checkpointing to save memory.
        
        Args:
            x: Input tensor [batch, seq_len, channels]
            mod: Modulation tensor [batch, channels]
            context: Context tensor for cross-attention [batch, context_len, ctx_channels]
            
        Returns:
            Processed tensor with same shape as input
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
class PartBased_ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    
    This block extends the standard transformer block with an additional cross-attention
    layer, allowing it to attend to a separate context input.
    """
    def __init__(
        self,
        channels: int,           # Number of input/output channels
        ctx_channels: int,       # Number of context channels
        num_heads: int,          # Number of attention heads
        mlp_ratio: float = 4.0,  # Ratio determining MLP hidden dimension size
        attn_mode: Literal["full", "windowed"] = "full",  # Attention computation mode
        window_size: Optional[int] = None,    # Size of attention window if using windowed attention
        shift_window: Optional[Tuple[int, int, int]] = None,  # Parameters for shifted window attention
        use_checkpoint: bool = False,  # Whether to use gradient checkpointing to save memory
        use_rope: bool = False,   # Whether to use Rotary Position Embedding
        qk_rms_norm: bool = False,  # Whether to use RMS normalization for query and key in self-attention
        qk_rms_norm_cross: bool = False,  # Whether to use RMS normalization for query and key in cross-attention
        qkv_bias: bool = True,    # Whether to use bias in QKV projection
        share_mod: bool = False,  # Whether to share modulation parameters externally
        is_obj_global_attn: bool = False,  # Whether to use object-global attention
        enable_global_attn_cross: bool = False,  # Whether to use object-global attention
        enable_cross_attn: bool = True,  # Whether to use object-global attention
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.is_obj_global_attn = is_obj_global_attn
        self.enable_global_attn_cross = enable_global_attn_cross
        self.enable_cross_attn = enable_cross_attn
        # print("is_obj_global_attn:",self.is_obj_global_attn)
        # Layer normalizations
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)  # For self-attention, will be modulated
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)  # For FFN, will be modulated
        
        # Self-attention layer
        self.self_attn = PartBased_MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            is_obj_global_attn=is_obj_global_attn,
            
        )
        
        # Cross-attention layer
        if self.enable_cross_attn:
            self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)  # For cross-attention, standard normalization
            
            self.cross_attn = PartBased_MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",  # Cross-attention always uses full attention
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
                is_obj_global_attn=enable_global_attn_cross and is_obj_global_attn,
                
            )
        
        # Feed-forward network
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        
        # Modulation network to generate adaptive parameters if not shared
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)  # 6 channels: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor,
                 num_parts:Optional[Union[int, torch.Tensor]] = None):
        """
        Internal forward function for the modulated transformer cross-attention block.
        
        Args:
            x: Input tensor [batch, seq_len, channels]
            mod: Modulation tensor [batch, channels]
            context: Context tensor for cross-attention [batch, context_len, ctx_channels]
            
        Returns:
            Processed tensor with same shape as input
        """
        # Split modulation vector into shift, scale, and gate parameters for MSA and FFN
        if self.share_mod:
            # Use externally provided modulation parameters
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            # Generate modulation parameters from the conditioning vector
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        
        # Apply modulated self-attention
        h = self.norm1(x)  # Normalize
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)  # Apply modulation
        if self.is_obj_global_attn:
            assert num_parts is not None, "num_parts must be provided for object-global attention"
        h = self.self_attn(h, num_parts=num_parts)  # Self-attention
        h = h * gate_msa.unsqueeze(1)  # Apply gate
        x = x + h  # Residual connection
        
        # Apply cross-attention (not modulated)
        if self.enable_cross_attn:
            h = self.norm2(x)  # Normalize
            h = self.cross_attn(h, context,num_parts=num_parts)  # Cross-attention with context
            x = x + h  # Residual connection
        
        # Apply modulated feed-forward network
        h = self.norm3(x)  # Normalize
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)  # Apply modulation
        h = self.mlp(h)  # Feed-forward
        h = h * gate_mlp.unsqueeze(1)  # Apply gate
        x = x + h  # Residual connection
        
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor,
                num_parts:Optional[Union[int, torch.Tensor]] = None
                ):
        """
        Forward pass with optional gradient checkpointing to save memory.
        
        Args:
            x: Input tensor [batch, seq_len, channels]
            mod: Modulation tensor [batch, channels]
            context: Context tensor for cross-attention [batch, context_len, ctx_channels]
            
        Returns:
            Processed tensor with same shape as input
        """
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, num_parts, use_reentrant=False)
        else:
            return self._forward(x, mod, context,num_parts)