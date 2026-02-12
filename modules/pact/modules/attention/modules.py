"""
This file contains attention mechanism implementations for the TRELLIS framework.
It provides various components needed for building transformer-based architectures,
including custom normalization, rotary position embeddings, and attention modules
with different configurations and optimizations.
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
from einops import rearrange, repeat

class MultiHeadRMSNorm(nn.Module):
    """
    Multi-head RMS normalization layer that applies per-head normalization.
    This helps stabilize attention computations by normalizing query and key vectors.
    
    Args:
        dim (int): The dimensionality of each head
        heads (int): Number of attention heads
    """
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization along the last dimension.
        
        Args:
            x (torch.Tensor): Input tensor of shape [..., dim]
            
        Returns:
            torch.Tensor: Normalized tensor with the same shape
        """
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    """
    Implements Rotary Position Embedding (RoPE), which encodes position information
    into the query and key tensors through a rotation-based approach.
    
    Args:
        hidden_size (int): Size of the hidden dimension
        in_channels (int): Number of input channels, defaults to 3
    """
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        # Calculate frequency bands on a log scale
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute phase shifts based on position indices.
        
        Args:
            indices (torch.Tensor): Position indices
            
        Returns:
            torch.Tensor: Complex tensor containing phase information
        """
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor
            phases (torch.Tensor): Phase tensor from _get_phases
            
        Returns:
            torch.Tensor: Tensor with rotary embeddings applied
        """
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q (torch.Tensor): [..., N, D] tensor of queries
            k (torch.Tensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions. If None,
                                   sequential indices will be used.
                                   
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Position-encoded query and key tensors
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed
    

class MultiHeadAttention(nn.Module):
    """
    Flexible multi-head attention implementation supporting both self-attention
    and cross-attention with various optimizations.
    
    Args:
        channels (int): Number of input/output channels
        num_heads (int): Number of attention heads
        ctx_channels (Optional[int]): Number of context channels for cross-attention
        type (str): Type of attention, either "self" or "cross"
        attn_mode (str): Attention computation mode, either "full" or "windowed"
        window_size (Optional[int]): Size of attention window if windowed mode is used
        shift_window (Optional[Tuple[int, int, int]]): Shift amount for windowed attention
        qkv_bias (bool): Whether to include bias in QKV projections
        use_rope (bool): Whether to use rotary position embeddings
        qk_rms_norm (bool): Whether to apply RMS normalization to Q and K
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        # Initialize projection layers based on attention type
        if self._type == "self":
            # For self-attention, create a single QKV projection
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            # For cross-attention, create separate projections for query and key-value
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        # Optional RMS normalization for stabilizing attention
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        # Output projection
        self.to_out = nn.Linear(channels, channels)

        # Optional rotary position embeddings
        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, C]
            context (Optional[torch.Tensor]): Context tensor for cross-attention
            indices (Optional[torch.Tensor]): Position indices for rotary embeddings
            
        Returns:
            torch.Tensor: Output tensor of shape [B, L, C]
        """
        B, L, C = x.shape
        if self._type == "self":
            # Self-attention path
            qkv = self.to_qkv(x) 
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1) ## torch.Size([1, 4096, 3, 16, 64])
            if self.use_rope:
                # Apply rotary position embeddings if enabled
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    # Apply RMS normalization to queries and keys if enabled
                    q, k, v = qkv.unbind(dim=2) ## torch.Size([1, 4096, 16, 64])
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v) ## torch.Size([1, 4096, 16, 64])
                else:
                    # Standard attention with combined QKV tensor
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:

            # Cross-attention path
            Lkv = context.shape[1]
            q = self.to_q(x)
            # print(f"context shape: {context.shape}")
            kv = self.to_kv(context)
            # print("reshape kv")
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            # print("unbind kv")
            if self.qk_rms_norm:
                # print("qk_rms_norm")
                # Apply RMS normalization to queries and keys if enabled
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                # print("unbind kv2")
                k = self.k_rms_norm(k)
                # print("unbind kv3")
                h = scaled_dot_product_attention(q, k, v)
                # print("unbind kv4")
            else:
                # Standard cross-attention
                # print("unbind kv2")
                # print(kv.shape)
                h = scaled_dot_product_attention(q, kv)
                # print("unbind kv3")
        # Reshape and project back to the original dimension
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h
    
class PartBased_MultiHeadAttention(nn.Module):
    """
    Flexible multi-head attention implementation supporting both self-attention
    and cross-attention with various optimizations.
    
    Args:
        channels (int): Number of input/output channels
        num_heads (int): Number of attention heads
        ctx_channels (Optional[int]): Number of context channels for cross-attention
        type (str): Type of attention, either "self" or "cross"
        attn_mode (str): Attention computation mode, either "full" or "windowed"
        window_size (Optional[int]): Size of attention window if windowed mode is used
        shift_window (Optional[Tuple[int, int, int]]): Shift amount for windowed attention
        qkv_bias (bool): Whether to include bias in QKV projections
        use_rope (bool): Whether to use rotary position embeddings
        qk_rms_norm (bool): Whether to apply RMS normalization to Q and K
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        is_obj_global_attn: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        self.is_obj_global_attn = is_obj_global_attn
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        # Initialize projection layers based on attention type
        if self._type == "self":
            # For self-attention, create a single QKV projection
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            # For cross-attention, create separate projections for query and key-value
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        # Optional RMS normalization for stabilizing attention
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        # Output projection
        self.to_out = nn.Linear(channels, channels)

        # Optional rotary position embeddings
        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                indices: Optional[torch.Tensor] = None,
                num_parts:Optional[Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        Apply multi-head attention to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, C]
            context (Optional[torch.Tensor]): Context tensor for cross-attention
            indices (Optional[torch.Tensor]): Position indices for rotary embeddings
            
        Returns:
            torch.Tensor: Output tensor of shape [B, L, C]
        """
        B, L, C = x.shape
        if self._type == "self":
            # Self-attention path
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                # Apply rotary position embeddings if enabled
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.is_obj_global_attn:
                if self.attn_mode != "full":
                    raise NotImplementedError("Object global attention currently supports full attention only")

                q, k, v = qkv.unbind(dim=2)
                if self.qk_rms_norm:
                    # Apply RMS normalization to queries and keys if enabled
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)

                h = self._apply_obj_global_attention(q, k, v, num_parts)
            else:

                if self.attn_mode == "full":
                    if self.qk_rms_norm:
                        # Apply RMS normalization to queries and keys if enabled
                        q, k, v = qkv.unbind(dim=2)
                        q = self.q_rms_norm(q)
                        k = self.k_rms_norm(k)
                        h = scaled_dot_product_attention(q, k, v)
                    else:
                        # Standard attention with combined QKV tensor
                        h = scaled_dot_product_attention(qkv)
                elif self.attn_mode == "windowed":
                    raise NotImplementedError("Windowed attention is not yet implemented")
        else:## "cross_attention"
            if self.is_obj_global_attn:
                # context.split(num_parts.tolist(),dim=0)
                first_idx = torch.cumsum(torch.cat([torch.tensor([0],device=context.device),
                                                    num_parts]),dim=0)
                context_chunks = [context[first_idx[i]:first_idx[i]+1] for i in range(len(num_parts))]
                context_objs = torch.cat(context_chunks,dim=0)  ## torch.Size([1, num_objects, 1024])
                Lkv = context.shape[1] ## torch.Size([11, 1374, 1024])
                kv = self.to_kv(context_objs)
                q = self.to_q(x)
                # print(f"context shape: {context.shape}")
                q = q.reshape(B, L, self.num_heads, -1)
                B_obj = num_parts.shape[0]
                
                kv = kv.reshape(B_obj, Lkv, 2, self.num_heads, -1)
                h_s = []
                for part_idx in range(len(num_parts)):
                    
                    q_i  = q[first_idx[part_idx]:first_idx[part_idx+1]]  ## [b*ni, L, h, c]
                    q_i = rearrange(
                            q_i, "(b ni) nt h c -> b (ni nt) h c", ni=num_parts[part_idx]
                        ) # [b, ni*nt, h, c]
                    kv_i = kv[part_idx:part_idx+1]
                    if self.qk_rms_norm:
                        raise NotImplementedError("Object global attention with RMS normalization is not implemented for cross-attention yet")
                    else:
                        h_i = scaled_dot_product_attention(q_i, kv_i)
                        h_i = rearrange(
                                    h_i, "b (ni nt) h c -> (b ni) nt h c", ni=num_parts[part_idx]
                                ) # [b*ni, nt, h, c]
                    h_s.append(h_i)     
                    
                h = torch.cat(h_s,dim=0)
                    
                raise NotImplementedError("Object global attention is not implemented for cross-attention yet")
            else:
                # Cross-attention path
                Lkv = context.shape[1] ## torch.Size([11, 1374, 1024])
                q = self.to_q(x)
                # print(f"context shape: {context.shape}")
                kv = self.to_kv(context)
                # print("reshape kv")
                q = q.reshape(B, L, self.num_heads, -1)
                kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
                # print("unbind kv")
                if self.qk_rms_norm:
                    # print("qk_rms_norm")
                    # Apply RMS normalization to queries and keys if enabled
                    q = self.q_rms_norm(q)
                    k, v = kv.unbind(dim=2)
                    # print("unbind kv2")
                    k = self.k_rms_norm(k)
                    # print("unbind kv3")
                    h = scaled_dot_product_attention(q, k, v)
                    # print("unbind kv4")
                else:
                    # Standard cross-attention
                    # print("unbind kv2")
                    # print(kv.shape)
                    h = scaled_dot_product_attention(q, kv)
                    # print("unbind kv3")
        # Reshape and project back to the original dimension
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h

    def _format_num_parts(self, num_parts: Optional[Union[int, torch.Tensor]], total_parts: int, device: torch.device) -> torch.Tensor:
        """Validate and normalize num_parts into a 1D LongTensor."""
        if num_parts is None:
            raise ValueError("num_parts must be provided when object-global attention is enabled")

        if isinstance(num_parts, int):
            assert total_parts % num_parts == 0, (
                f"total parts ({total_parts}) must be divisible by num_parts ({num_parts}) when provided as int"
            )
            normalized = torch.full(
                (total_parts // num_parts,), num_parts, device=device, dtype=torch.long
            )
        elif isinstance(num_parts, torch.Tensor):
            normalized = num_parts.to(device=device, dtype=torch.long).flatten()
        else:
            raise ValueError("num_parts must be either int or torch.Tensor")

        assert (
            int(normalized.sum().item()) == total_parts
        ), f"Sum of num_parts ({normalized.sum().item()}) must match total parts ({total_parts})"
        return normalized

    def _apply_obj_global_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_parts: Optional[Union[int, torch.Tensor]],
    ) -> torch.Tensor:
        """Run attention independently for each object's parts."""
        part_sizes = self._format_num_parts(num_parts, q.shape[0], q.device)
        if part_sizes.numel() == 0:
            return q.new_zeros(q.shape)

        # Prepare split points once to avoid slicing in Python loops
        split_indices = part_sizes.cumsum(0)[:-1].tolist()
        q_chunks = torch.tensor_split(q, split_indices, dim=0)
        k_chunks = torch.tensor_split(k, split_indices, dim=0)
        v_chunks = torch.tensor_split(v, split_indices, dim=0)

        outputs: List[torch.Tensor] = []
        for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
            # Flatten part dimension into the token dimension for attention
            num_part_tokens = q_chunk.shape[0] * q_chunk.shape[1]
            q_obj = q_chunk.reshape(1, num_part_tokens, self.num_heads, self.head_dim)
            k_obj = k_chunk.reshape(1, num_part_tokens, self.num_heads, self.head_dim)
            v_obj = v_chunk.reshape(1, num_part_tokens, self.num_heads, self.head_dim)

            h_obj = scaled_dot_product_attention(q_obj, k_obj, v_obj)
            outputs.append(h_obj.reshape_as(q_chunk))

        result = torch.cat(outputs, dim=0)
        assert result.shape == q.shape, "Object attention must preserve the per-part layout"
        return result
