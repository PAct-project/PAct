"""directly copy from singapo https://github.com/3dlg-hcvc/singapo/blob/8c4cbf63938e88c9a221abce09e5af00355ba12e/models/denoiser.py#L246
    and  https://github.com/3dlg-hcvc/singapo/blob/8c4cbf63938e88c9a221abce09e5af00355ba12e/models/utils.py#L4

Returns:
    _type_: _description_
"""
import os, sys
import torch
import models
from torch import nn
from diffusers.models.attention import Attention, FeedForward
import torch
from torch import nn
from typing import Optional
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, LabelEmbedding

class FinalLayer(nn.Module):
    """
    Final layer of the diffusion model that outputs the final logits.
    """
    def __init__(self, in_ch, out_ch=None, dropout=0.0):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.linear = nn.Linear(in_ch, out_ch)
        self.norm = AdaLayerNormTC(in_ch, 2 * in_ch, dropout)

    def forward(self, x, t, cond=None):
        assert cond is not None
        x = self.norm(x, t, cond)
        x = self.linear(x)
        return x


class AdaLayerNormTC(nn.Module):
    """
    Norm layer modified to incorporate timestep and condition embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings, dropout):
        super().__init__()
        self.emb = CombinedTimestepLabelEmbeddings(
            num_embeddings, embedding_dim, dropout
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(
            embedding_dim, elementwise_affine=False, eps=torch.finfo(torch.float16).eps
        )

    def forward(self, x, timestep, cond):
        emb = self.linear(self.silu(self.emb(timestep, cond, hidden_dtype=None)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class PEmbeder(nn.Module):
    """
    Positional embedding layer.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in")

    def forward(self, x, idx=None):
        if idx is None:
            idx = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        return x + self.embed(idx)

class CombinedTimestepLabelEmbeddings(nn.Module):
    '''Modified from diffusers.models.embeddings.CombinedTimestepLabelEmbeddings'''
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)

    def forward(self, timestep, class_labels, hidden_dtype=None, label_free=False):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        force_drop_ids = None # training mode
        if label_free: # inference mode, force_drop_ids is set to all ones to be dropped in class_embedder
            force_drop_ids = torch.ones_like(class_labels, dtype=torch.bool, device=class_labels.device)
        class_labels = self.class_embedder(class_labels, force_drop_ids)  # (N, D)
        conditioning = timesteps_emb + class_labels  # (N, D)
        return conditioning


class MyAdaLayerNormZero(nn.Module):
    """
    Adaptive layer norm zero (adaLN-Zero), borrowed from diffusers.models.attention.AdaLayerNormZero.
    Extended to incorporate scale parameters (gate_2, gate_3) for intermidate attention layers.
    """

    def __init__(self, embedding_dim, num_embeddings, class_dropout_prob):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(
            num_embeddings, embedding_dim, class_dropout_prob
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 8 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep, class_labels, hidden_dtype=None, label_free=False):
        emb_t_cls = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype, label_free=label_free)
        emb = self.linear(self.silu(emb_t_cls))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            gate_2,
            gate_3,
        ) = emb.chunk(8, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_2, gate_3


class VisAttnProcessor:
    r"""
    This code is adapted from diffusers.models.attention_processor.AttnProcessor.
    Used for visualizing the attention maps.
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Removed
        # if len(args) > 0 or kwargs.get("scale", None) is not None:
        #     deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        #     deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query) # (40, 160, 16)
        key = attn.head_to_batch_dim(key) # (40, 256, 16)
        value = attn.head_to_batch_dim(value)  # (40, 256, 16)
        
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(attention_mask, dtype=query.dtype, device=query.device)
                attn_mask = attn_mask.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attn_mask = attention_mask
                assert attn_mask.dtype == query.dtype, f"query and attention_mask must have the same dtype, but got {query.dtype} and {attention_mask.dtype}."
        else:
            attn_mask = None
        attention_probs = attn.get_attention_scores(query, key, attn_mask) # (40, 160, 256)
        hidden_states = torch.bmm(attention_probs, value) # (40, 160, 16)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        attention_probs = attention_probs.reshape(batch_size, attn.heads, query.shape[1], sequence_length)

        return hidden_states, attention_probs



class Attn_Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
        class_dropout_prob: float = 0.0,  # for classifier-free
        img_emb_dims=None,

    ):
        super().__init__()

        self.norm1 = MyAdaLayerNormZero(dim, num_embeds_ada_norm, class_dropout_prob)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.norm4 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.norm5 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        self.local_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.global_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.graph_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.img_attn = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_norm="layer_norm",
            processor=VisAttnProcessor(), 
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # image embedding layers
        layers = []
        for i in range(len(img_emb_dims) - 1):
            layers.append(nn.Linear(img_emb_dims[i], img_emb_dims[i + 1]))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.pop(-1)
        self.img_emb = nn.Sequential(*layers)
        self.init_img_emb_weights()

    def init_img_emb_weights(self):
        for m in self.img_emb.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        hidden_states,
        img_patches,
        pad_mask,
        attr_mask,
        graph_mask,
        timestep,
        class_labels,
        label_free=False,
    ):
        # image patches embedding
        img_emb = self.img_emb(img_patches)

        # adaptive normalization, taken timestep and class_labels as input condition
        norm_hidden_states, gate_1, shift_mlp, scale_mlp, gate_mlp, gate_2, gate_3 = (
            self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype,
                label_free=label_free
            )
        )

        # local attribute self-attention
        attr_out = self.local_attn(norm_hidden_states, attention_mask=attr_mask)
        attr_out = gate_1.unsqueeze(1) * attr_out
        hidden_states = hidden_states + attr_out

        # global attribute self-attention
        norm_hidden_states = self.norm2(hidden_states)
        global_out = self.global_attn(norm_hidden_states, attention_mask=pad_mask)
        global_out = gate_2.unsqueeze(1) * global_out
        hidden_states = hidden_states + global_out

        # graph relation self-attention
        norm_hidden_states = self.norm3(hidden_states)
        graph_out = self.graph_attn(norm_hidden_states, attention_mask=graph_mask)
        graph_out = gate_3.unsqueeze(1) * graph_out
        hidden_states = hidden_states + graph_out

        # cross attention with image patches
        norm_hidden_states = self.norm4(hidden_states)
        B, Na, D = norm_hidden_states.shape
        Np = img_emb.shape[1] # number of image patches
        reshaped = norm_hidden_states.reshape(B, Na // 5, 5, D)
        bboxes = reshaped[:, :, 0, :] # (B, K, D)
        # cross attention between bbox attributes and image patches
        bbox_img_out, bbox_cross_attn_map = self.img_attn(
            bboxes,
            encoder_hidden_states=img_emb,
            attention_mask=None,
        )  # cross_attn_map: (B, n_head, K, Np)

        # to reshape the cross_attn_map back to (B, n_head, Na*5, Np), reduntant for other attributes, fix later
        cross_attn_map_reshape = torch.zeros(size=(B, bbox_cross_attn_map.shape[1], Na // 5, 5, Np), device=bbox_cross_attn_map.device)
        cross_attn_map_reshape[:, :, :, 0, :] = bbox_cross_attn_map
        cross_attn_map = cross_attn_map_reshape.reshape(B, bbox_cross_attn_map.shape[1], Na, Np)

        # assemble the output of cross attention with bbox attributes and other attributes
        img_out = torch.empty(size=(B, Na // 5, 5, D), device=hidden_states.device, dtype=hidden_states.dtype)
        img_out[:, :, 0, :] = bbox_img_out
        img_out[:, :, 1:, :] = reshaped[:, :, 1:, :]
        img_out = img_out.reshape(B, Na, D)
        hidden_states = hidden_states + img_out

        # feed-forward
        norm_hidden_states = self.norm5(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = ff_output + hidden_states

        return hidden_states, cross_attn_map


@models.register("denoiser")
class Denoiser(nn.Module):
    """
    Denoiser based on CAGE's attribute attention block + our ICA module, with 4 sequential attentions: LA -> GA -> GRA -> ICA
    Different image adapters for each layer.
    The image cross attention is with key-padding masks (object mask, part mask)
    *** The ICA only applies to the bbox attributes, not other attributes such as motion params.***
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.K = self.hparams.get("K", 32)

        in_ch = hparams.in_ch
        attn_dim = hparams.attn_dim
        mid_dim = attn_dim // 2
        n_head = hparams.n_head
        head_dim = attn_dim // n_head
        num_embeds_ada_norm = 6 * attn_dim
        
        # embedding layers for different node attributes
        self.aabb_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.jaxis_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.range_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.label_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.jtype_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        # positional encoding for nodes and attributes
        self.pe_node = PEmbeder(self.K, attn_dim)
        self.pe_attr = PEmbeder(5, attn_dim)

        # attention layers
        self.attn_layers = nn.ModuleList(
            [
                Attn_Block(
                    dim=attn_dim,
                    num_attention_heads=n_head,
                    attention_head_dim=head_dim,
                    class_dropout_prob=hparams.get("cat_drop_prob", 0.0),
                    dropout=hparams.dropout,
                    activation_fn="geglu",
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=False,
                    norm_elementwise_affine=True,
                    final_dropout=False,
                    img_emb_dims=hparams.get("img_emb_dims", None),
                )
                for d in range(hparams.n_layers)
            ]
        )

        self.final_layer = FinalLayer(attn_dim, in_ch)

    def forward(
        self,
        x,
        cat,
        timesteps,
        feat,
        key_pad_mask=None,
        graph_mask=None,
        attr_mask=None,
        label_free=False,
    ):
        B = x.shape[0]
        x = x.view(B, self.K, 5 * 6)

        # embedding layers for different attributes
        x_aabb = self.aabb_emb(x[..., :6])
        x_jtype = self.jtype_emb(x[..., 6:12])
        x_jaxis = self.jaxis_emb(x[..., 12:18])
        x_range = self.range_emb(x[..., 18:24])
        x_label = self.label_emb(x[..., 24:30])

        # concatenate all attribute embeddings
        x_ = torch.cat(
            [x_aabb, x_jtype, x_jaxis, x_range, x_label], dim=2
        )  # (B, K, 5*attn_dim)
        x_ = x_.view(B, self.K * 5, self.hparams.attn_dim)

        # positional encoding for nodes and attributes
        idx_attr = torch.tensor(
            [0, 1, 2, 3, 4], device=x.device, dtype=torch.long
        ).repeat(self.K)
        idx_node = torch.arange(
            self.K, device=x.device, dtype=torch.long
        ).repeat_interleave(5)
        x_ = self.pe_attr(self.pe_node(x_, idx=idx_node), idx=idx_attr)


        # init tensor to store attention maps
        Np = feat.shape[1]
        attn_maps = torch.empty(
            size=(B * self.hparams.n_layers, self.hparams.n_head, self.K*5, Np),
            device=x.device,
        )

        # attention layers
        for i, attn_layer in enumerate(self.attn_layers):
            x_, attn_map = attn_layer(
                hidden_states=x_,
                img_patches=feat,
                timestep=timesteps,
                class_labels=cat,
                pad_mask=key_pad_mask,
                graph_mask=graph_mask,
                attr_mask=attr_mask,
                label_free=label_free,
            )
            # store attention maps
            attn_maps[i * B : i * B + B] = attn_map

        y = self.final_layer(x_, timesteps, cat)
        return {
            'noise_pred': y,
            'attn_maps': attn_maps,
        }

class Denoiser_Head_QML(nn.Module):


    def __init__(self,                     
                    out_channel=24,
                    arti_token_length =3,
                    model_channels=1024,
                    cond_channels=1024,
                    num_blocks=1,
                    mlp_ratio: float = 4,
                    use_checkpoint: bool = False,
                    share_mod: bool = False,
                    num_heads: Optional[int] = 4,
                    part_feat_channel =1024,
                    enable_cross_attn: bool = False, ### False for SS_adapter
                    use_fp16: bool = True,
                    ):
        super().__init__()
        self.K = 32

        in_ch = hparams.in_ch
        attn_dim = hparams.attn_dim
        mid_dim = attn_dim // 2
        n_head = hparams.n_head
        head_dim = attn_dim // n_head
        num_embeds_ada_norm = 6 * attn_dim
        
        # embedding layers for different node attributes
        self.aabb_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.jaxis_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.range_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.label_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.jtype_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        # positional encoding for nodes and attributes
        self.pe_node = PEmbeder(self.K, attn_dim)
        self.pe_attr = PEmbeder(5, attn_dim)

        # attention layers
        self.attn_layers = nn.ModuleList(
            [
                Attn_Block(
                    dim=attn_dim,
                    num_attention_heads=n_head,
                    attention_head_dim=head_dim,
                    class_dropout_prob=hparams.get("cat_drop_prob", 0.0),
                    dropout=hparams.dropout,
                    activation_fn="geglu",
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=False,
                    norm_elementwise_affine=True,
                    final_dropout=False,
                    img_emb_dims=hparams.get("img_emb_dims", None),
                )
                for d in range(hparams.n_layers)
            ]
        )

        self.final_layer = FinalLayer(attn_dim, in_ch)

    def forward(
        self,
        x,
        cat,
        timesteps,
        feat,
        key_pad_mask=None,
        graph_mask=None,
        attr_mask=None,
        label_free=False,
    ):
        B = x.shape[0]
        x = x.view(B, self.K, 5 * 6)

        # embedding layers for different attributes
        x_aabb = self.aabb_emb(x[..., :6])
        x_jtype = self.jtype_emb(x[..., 6:12])
        x_jaxis = self.jaxis_emb(x[..., 12:18])
        x_range = self.range_emb(x[..., 18:24])
        x_label = self.label_emb(x[..., 24:30])

        # concatenate all attribute embeddings
        x_ = torch.cat(
            [x_aabb, x_jtype, x_jaxis, x_range, x_label], dim=2
        )  # (B, K, 5*attn_dim)
        x_ = x_.view(B, self.K * 5, self.hparams.attn_dim)

        # positional encoding for nodes and attributes
        idx_attr = torch.tensor(
            [0, 1, 2, 3, 4], device=x.device, dtype=torch.long
        ).repeat(self.K)
        idx_node = torch.arange(
            self.K, device=x.device, dtype=torch.long
        ).repeat_interleave(5)
        x_ = self.pe_attr(self.pe_node(x_, idx=idx_node), idx=idx_attr)


        # init tensor to store attention maps
        Np = feat.shape[1]
        attn_maps = torch.empty(
            size=(B * self.hparams.n_layers, self.hparams.n_head, self.K*5, Np),
            device=x.device,
        )

        # attention layers
        for i, attn_layer in enumerate(self.attn_layers):
            x_, attn_map = attn_layer(
                hidden_states=x_,
                img_patches=feat,
                timestep=timesteps,
                class_labels=cat,
                pad_mask=key_pad_mask,
                graph_mask=graph_mask,
                attr_mask=attr_mask,
                label_free=label_free,
            )
            # store attention maps
            attn_maps[i * B : i * B + B] = attn_map

        y = self.final_layer(x_, timesteps, cat)
        return {
            'noise_pred': y,
            'attn_maps': attn_maps,
        }
