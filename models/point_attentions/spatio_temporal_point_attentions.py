# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from re import T
import ipdb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from einops import rearrange, repeat

from ..encoders.dinov2 import LayerScale, Mlp
from ..utils.self_cross_attention_layers import FFNLayer, KNNSelfAttentionLayer, SelfAttentionLayer


st = ipdb.set_trace
# pyre-strict

class SpatioTemporalPointAttentionsv2(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        nheads: int = 8,
        dropout: float = 0.0,
        num_layers: int = 6,
        knn_spatial_attn: bool = False,
        k: int = 16,
    ) -> None:
        super().__init__()
        # self.pe_layer = PositionEmbeddingLearned(dim=3, num_pos_feats=latent_dim)
        self.num_layers = num_layers

        if knn_spatial_attn:
            self.self_attention_layers_spatial = nn.ModuleList(
                [
                    KNNSelfAttentionLayer(
                        d_model=latent_dim,
                        nhead=nheads,
                        dropout=dropout,
                        k=k,
                        normalize_before=True,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.self_attention_layers_spatial = nn.ModuleList(
                [
                    SelfAttentionLayer(
                        d_model=latent_dim,
                        nhead=nheads,
                        dropout=dropout,
                        activation="relu",
                        normalize_before=True,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.self_attention_layers_temporal = nn.ModuleList(
            [
                SelfAttentionLayer(
                    d_model=latent_dim,
                    nhead=nheads,
                    dropout=dropout,
                    activation="relu",
                    normalize_before=True,
                )
                for _ in range(num_layers)
            ]
        )

        dim_feedforward = 4 * latent_dim

        self.spatial_ffn_layers = nn.ModuleList(
            [
                FFNLayer(
                    d_model=latent_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=True,
                    activation="relu",
                )
                for _ in range(num_layers)
            ]
        )
        self.temporal_ffn_layers = nn.ModuleList(
            [
                FFNLayer(
                    d_model=latent_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=True,
                    activation="relu",
                )
                for _ in range(num_layers)
            ]
        )
        # self.layer_norm_static = nn.LayerNorm(latent_dim)
        # self.layer_norm_dynamic = nn.LayerNorm(latent_dim)
        self.layer_norms_static = nn.ModuleList(
            [nn.LayerNorm(latent_dim) for _ in range(num_layers)]
        )
        self.layer_norms_dynamic = nn.ModuleList(
            [nn.LayerNorm(latent_dim) for _ in range(num_layers)]
        )
        self.static_dynamic = nn.Embedding(2, latent_dim)

        # self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

    def forward(
        self,
        static_output: torch.Tensor,
        dynamic_output: torch.Tensor,
        static_pos_embeds: torch.Tensor,
        dynamic_pos_embeds: torch.Tensor,
        temporal_cls_token_feat: torch.Tensor,
        temporal_cls_token_pos: torch.Tensor,
        knn_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        input:
            point_features: [B, N, C]
            pos3d_time_embeds: [B, N, T, C]
            gt_is_dynamic: [B, N]
            points: [B, N, 3]
            knn_indices: [B, N, K]
        return: [B, N, T, C]
        """
        # output = point_features
        # temporal_cls_token_feat = dynamic_output.mean(1)
        # temporal_cls_token_pos = dynamic_pos_embeds.mean(1)
        static_output = static_output + static_pos_embeds + self.static_dynamic.weight[0][None].repeat(
                    static_pos_embeds.shape[0], 1
                )
        temporal_cls_token_feat = temporal_cls_token_feat + temporal_cls_token_pos + self.static_dynamic.weight[1][None].repeat(
                    temporal_cls_token_pos.shape[0], 1
                )
        dynamic_output = dynamic_output + dynamic_pos_embeds
        
        for i in range(self.num_layers):
            spatial_output = self.self_attention_layers_spatial[i](
                torch.cat([static_output, temporal_cls_token_feat], dim=0)[
                    None
                ].permute(1, 0, 2),
                query_pos=None,
                knn_indices=knn_indices,
            ).permute(1, 0, 2)[0]
            spatial_output = self.spatial_ffn_layers[i](spatial_output)
            spatial_output = self.layer_norms_static[i](spatial_output)
            static_output = spatial_output[: static_output.shape[0]]
            temporal_cls_token_feat = spatial_output[static_output.shape[0] :]

            # Apply temporal attention only to dynamic points
            if dynamic_output.shape[0] != 0:
                dynamic_output = torch.cat(
                    [dynamic_output, temporal_cls_token_feat[:, None]], dim=1
                )
                seq_len = dynamic_output.shape[0]
                max_chunk = 60000  # FlashAttn safe upper bound (~64K)

                if seq_len <= max_chunk:
                    # Normal case — no split
                    dynamic_output = self.self_attention_layers_temporal[i](
                        dynamic_output.permute(1, 0, 2),
                        query_pos=None,
                    ).permute(1, 0, 2)
                else:
                    # Chunked case — split sequence dimension
                    chunks = []
                    for start in range(0, seq_len, max_chunk):
                        end = min(start + max_chunk, seq_len)
                        out_chunk = self.self_attention_layers_temporal[i](
                            dynamic_output[start:end].permute(1, 0, 2),
                            query_pos=None,
                        ).permute(1, 0, 2)
                        chunks.append(out_chunk)
                    dynamic_output = torch.cat(chunks, dim=0)

                dynamic_output = self.temporal_ffn_layers[i](dynamic_output)
                dynamic_output = self.layer_norms_dynamic[i](dynamic_output)

                temporal_cls_token_feat = dynamic_output[:, -1]
                dynamic_output = dynamic_output[:, :-1]
  
        return static_output, dynamic_output, temporal_cls_token_feat


class FlashSelfAttention(nn.Module):
    """Self-attention using F.scaled_dot_product_attention (flash / efficient backend).
    Weight-compatible with DINOv2's Attention / MemEffAttention (same qkv + proj layout)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop_p if self.training else 0.0,
            )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DinoTransformerBlock(nn.Module):
    """Pre-norm transformer block with the same parameter layout as DINOv2's Block /
    NestedTensorBlock so that pretrained weights can be loaded with load_state_dict.

    Uses FlashSelfAttention instead of xformers MemEffAttention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float = 1.0,
        act_layer: type = nn.GELU,
    ) -> None:        
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = FlashSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class FlashCrossAttention(nn.Module):
    """Cross-attention weight-compatible with DINOv2's Attention (same ``qkv`` + ``proj``
    parameter layout).

    The combined ``qkv`` Linear is reused at runtime: its first ``dim`` output rows act
    as ``W_q`` (applied to the query input ``x``), and the remaining ``2 * dim`` rows act
    as ``W_kv`` (applied to ``context``). This way a DINOv2 self-attention ``state_dict``
    loads directly, and when ``context is None`` (or ``context == x``) the module
    reproduces self-attention exactly.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            context = x
        B, N, C = x.shape
        _, M, _ = context.shape
        D = self.dim

        w = self.qkv.weight
        w_q, w_kv = w[:D], w[D:]
        if self.qkv.bias is not None:
            b = self.qkv.bias
            b_q, b_kv = b[:D], b[D:]
        else:
            b_q, b_kv = None, None

        q = F.linear(x, w_q, b_q)
        kv = F.linear(context, w_kv, b_kv)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = (
            kv.reshape(B, M, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop_p if self.training else 0.0,
            )
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class DinoCrossTransformerBlock(nn.Module):
    """Pre-norm cross-attention block with a parameter layout that mirrors DINOv2's
    Block (``norm1``, ``attn.qkv``, ``attn.proj``, ``ls1``, ``norm2``, ``mlp.*``,
    ``ls2``), plus a separate ``norm_context`` LayerNorm applied to the key/value
    stream.

    Pretrained DINOv2 block weights can be loaded via :meth:`load_dino_block_state_dict`,
    which copies ``norm1`` into ``norm_context`` so that when ``context is None`` /
    ``context == x`` the block is numerically equivalent to :class:`DinoTransformerBlock`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float = 1.0,
        act_layer: type = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm_context = nn.LayerNorm(dim, eps=1e-6)
        self.attn = FlashCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm1(x)
        kv = q if context is None else self.norm_context(context)
        x = x + self.ls1(self.attn(q, kv))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

    @torch.no_grad()
    def load_dino_block_state_dict(
        self, state_dict: dict, strict: bool = True
    ) -> None:
        """Load a DINOv2 Block ``state_dict`` (no ``norm_context`` key) into this block.

        ``norm_context`` is initialized as a copy of ``norm1`` so that, absent any
        context input, this block matches the DINO self-attention block exactly.
        """
        sd = dict(state_dict)
        if "norm1.weight" in sd and "norm_context.weight" not in sd:
            sd["norm_context.weight"] = sd["norm1.weight"].clone()
        if "norm1.bias" in sd and "norm_context.bias" not in sd:
            sd["norm_context.bias"] = sd["norm1.bias"].clone()
        return self.load_state_dict(sd, strict=strict)


_DINOV2_VIT_PRETRAIN_URL = {
    "vitg14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
    "vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
}
_DINOV2_VIT_DEPTH = {
    "vitg14": 40,
    "vits14": 12,
    "vitb14": 12,
}


class SpatioTemporalPointAttentionsDino(nn.Module):
    """Spatio-temporal point attention built from DINOv2-compatible transformer blocks.

    Each of the ``num_layers`` iterations runs:
        1. A **spatial** DinoTransformerBlock over (static tokens + temporal cls tokens).
        2. A **temporal** DinoTransformerBlock over each dynamic point's time sequence
           (with the temporal cls token appended).

    Positional embeddings are added *once* before entering the layers (same
    pattern as SpatioTemporalPointAttentionsv2).

    Pretrained weights from the last ``num_layers * 2`` blocks of a DINOv2 ViT
    can be loaded via :meth:`load_dino_weights`.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        nheads: int = 6,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float = 1.0,
        dropout: float = 0.0,
        knn_spatial_attn: bool = False,
        k: int = 16,
        vit_name: str = "vits14",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.vit_name = vit_name
        
        if self.vit_name == "vits14":
            latent_dim = 384
            nheads = 6
        elif self.vit_name == "vitb14":
            latent_dim = 768
            nheads = 12
            num_layers = 3
            self.num_layers = 3
        elif self.vit_name == "vitl14":
            latent_dim = 1024
            nheads = 16
        elif self.vit_name == "vitg14":
            latent_dim = 1536
            nheads = 24
        
        block_kwargs = dict(
            dim=latent_dim,
            num_heads=nheads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
        )

        self.spatial_blocks = nn.ModuleList(
            [DinoTransformerBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self.temporal_blocks = nn.ModuleList(
            [DinoTransformerBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self.static_dynamic = nn.Embedding(2, latent_dim)
        self.norm_static = nn.LayerNorm(latent_dim, eps=1e-6)
        self.norm_dynamic = nn.LayerNorm(latent_dim, eps=1e-6)
        
        self.load_dino_weights()

    def load_dino_weights(
        self,
        state_dict: dict | None = None,
        dino_depth: int | None = None,
    ) -> None:
        """Load weights from a DINOv2 ViT state dict.

        The last ``num_layers * 2`` blocks are mapped to alternating
        spatial / temporal blocks::

            dino blocks[start + 2*i]     ->  spatial_blocks[i]
            dino blocks[start + 2*i + 1] ->  temporal_blocks[i]

        Args:
            state_dict: Full DINOv2 ViT state dict (keys like ``blocks.{i}.norm1.weight``),
                or ``None`` to download pretrained weights for ``self.vit_name`` (e.g.
                ``vits14`` or ``vitg14``) from the official DINOv2 URL.
            dino_depth: Total depth of the source ViT. Required when ``state_dict`` is
                provided (e.g. 12 for ViT-S/B, 24 for ViT-L, 40 for ViT-g/14). When
                ``state_dict`` is ``None``, depth follows the architecture (12 for
                ``vits14``, 40 for ``vitg14``).
        """
        if state_dict is None:
            try:
                url = _DINOV2_VIT_PRETRAIN_URL[self.vit_name]
                dino_depth = _DINOV2_VIT_DEPTH[self.vit_name]
            except KeyError as e:
                raise KeyError(
                    f"Unknown vit_name {self.vit_name!r}; expected one of "
                    f"{sorted(_DINOV2_VIT_PRETRAIN_URL.keys())}"
                ) from e
            state_dict = torch.hub.load_state_dict_from_url(
                url,
                map_location="cpu",
                progress=True,
            )
        elif dino_depth is None:
            raise ValueError(
                "dino_depth is required when state_dict is provided "
                "(e.g. 12, 24, or 40 for ViT-g/14)."
            )

        start_idx = dino_depth - self.num_layers * 2
        assert start_idx >= 0, (
            f"DINOv2 depth ({dino_depth}) too small for {self.num_layers} layer pairs "
            f"(need at least {self.num_layers * 2} blocks)"
        )

        for i in range(self.num_layers):
            spatial_prefix = f"blocks.{start_idx + 2 * i}."
            temporal_prefix = f"blocks.{start_idx + 2 * i + 1}."
            spatial_sd = {
                k[len(spatial_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(spatial_prefix)
            }
            temporal_sd = {
                k[len(temporal_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(temporal_prefix)
            }

            miss_s, unexp_s = self.spatial_blocks[i].load_state_dict(spatial_sd, strict=False)
            miss_t, unexp_t = self.temporal_blocks[i].load_state_dict(temporal_sd, strict=False)

            if miss_s:
                print(f"[load_dino_weights] spatial_blocks[{i}] missing keys: {miss_s}")
            if miss_t:
                print(f"[load_dino_weights] temporal_blocks[{i}] missing keys: {miss_t}")

        norm_sd = {
            k[len("norm."):]: v
            for k, v in state_dict.items()
            if k.startswith("norm.") and not k.startswith("norm_")
        }
        if norm_sd:
            self.norm_static.load_state_dict(norm_sd)
            self.norm_dynamic.load_state_dict(norm_sd)
            print("[load_dino_weights] Loaded DINOv2 final norm into norm_static and norm_dynamic")

    def forward(
        self,
        static_output: torch.Tensor,
        dynamic_output: torch.Tensor,
        static_pos_embeds: torch.Tensor,
        dynamic_pos_embeds: torch.Tensor,
        temporal_cls_token_feat: torch.Tensor,
        temporal_cls_token_pos: torch.Tensor,
        knn_indices: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            static_output:          [N_s, C]
            dynamic_output:         [N_d, T, C]
            static_pos_embeds:      [N_s, C]
            dynamic_pos_embeds:     [N_d, T, C]
            temporal_cls_token_feat:[N_d, C]
            temporal_cls_token_pos: [N_d, C]

        Returns:
            static_output:          [N_s, C]
            dynamic_output:         [N_d, T, C]
            temporal_cls_token_feat:[N_d, C]
        """
        static_output = (
            static_output
            + static_pos_embeds
            + self.static_dynamic.weight[0][None].expand(static_pos_embeds.shape[0], -1)
        )
        temporal_cls_token_feat = (
            temporal_cls_token_feat
            + temporal_cls_token_pos
            + self.static_dynamic.weight[1][None].expand(temporal_cls_token_pos.shape[0], -1)
        )
        dynamic_output = dynamic_output + dynamic_pos_embeds

        for i in range(self.num_layers):
            # --- Spatial attention (static tokens + temporal cls tokens) ---
            n_static = static_output.shape[0]
            spatial_input = torch.cat(
                [static_output, temporal_cls_token_feat], dim=0
            ).unsqueeze(0)  # [1, N_s + N_d, C]

            spatial_output = self.spatial_blocks[i](spatial_input).squeeze(0)
            static_output = spatial_output[:n_static]
            temporal_cls_token_feat = spatial_output[n_static:]

            # --- Temporal attention per dynamic point ---
            if dynamic_output.shape[0] != 0:
                dynamic_with_cls = torch.cat(
                    [dynamic_output, temporal_cls_token_feat[:, None]], dim=1
                )  # [N_d, T+1, C]

                n_points = dynamic_with_cls.shape[0]
                max_chunk = 60000

                if n_points <= max_chunk:
                    dynamic_with_cls = self.temporal_blocks[i](dynamic_with_cls)
                else:
                    chunks = []
                    for start in range(0, n_points, max_chunk):
                        end = min(start + max_chunk, n_points)
                        chunks.append(
                            self.temporal_blocks[i](dynamic_with_cls[start:end])
                        )
                    dynamic_with_cls = torch.cat(chunks, dim=0)

                temporal_cls_token_feat = dynamic_with_cls[:, -1]
                dynamic_output = dynamic_with_cls[:, :-1]

        static_output = self.norm_static(static_output)
        dynamic_output = self.norm_dynamic(dynamic_output)

        return static_output, dynamic_output, temporal_cls_token_feat



class SpatialPointAttentionsDino(nn.Module):
    """Spatial point attention built from DINOv2-compatible transformer blocks.

    Each of the ``num_layers`` iterations runs:
        1. A **spatial** DinoTransformerBlock over all points.

    Positional embeddings are added *once* before entering the layers.

    Pretrained weights from the last ``num_layers`` blocks of a DINOv2 ViT
    can be loaded via :meth:`load_dino_weights`.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        nheads: int = 6,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float = 1.0,
        dropout: float = 0.0,
        knn_spatial_attn: bool = False,
        k: int = 16,
        vit_name: str = "vits14",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.vit_name = vit_name
        
        if self.vit_name == "vits14":
            latent_dim = 384
            nheads = 6
        elif self.vit_name == "vitb14":
            latent_dim = 768
            nheads = 12
            num_layers = 3
            self.num_layers = 3
        elif self.vit_name == "vitl14":
            latent_dim = 1024
            nheads = 16
        elif self.vit_name == "vitg14":
            latent_dim = 1536
            nheads = 24
        
        block_kwargs = dict(
            dim=latent_dim,
            num_heads=nheads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
        )

        self.spatial_blocks = nn.ModuleList(
            [DinoTransformerBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(latent_dim, eps=1e-6)
        
        self.load_dino_weights()

    def load_dino_weights(
        self,
        state_dict: dict | None = None,
        dino_depth: int | None = None,
    ) -> None:
        """Load weights from a DINOv2 ViT state dict.

        The last ``num_layers * 2`` blocks are mapped to alternating
        spatial / temporal blocks::

            dino blocks[start + 2*i]     ->  spatial_blocks[i]
            dino blocks[start + 2*i + 1] ->  temporal_blocks[i]

        Args:
            state_dict: Full DINOv2 ViT state dict (keys like ``blocks.{i}.norm1.weight``),
                or ``None`` to download pretrained weights for ``self.vit_name`` (e.g.
                ``vits14`` or ``vitg14``) from the official DINOv2 URL.
            dino_depth: Total depth of the source ViT. Required when ``state_dict`` is
                provided (e.g. 12 for ViT-S/B, 24 for ViT-L, 40 for ViT-g/14). When
                ``state_dict`` is ``None``, depth follows the architecture (12 for
                ``vits14``, 40 for ``vitg14``).
        """
        if state_dict is None:
            try:
                url = _DINOV2_VIT_PRETRAIN_URL[self.vit_name]
                dino_depth = _DINOV2_VIT_DEPTH[self.vit_name]
            except KeyError as e:
                raise KeyError(
                    f"Unknown vit_name {self.vit_name!r}; expected one of "
                    f"{sorted(_DINOV2_VIT_PRETRAIN_URL.keys())}"
                ) from e
            state_dict = torch.hub.load_state_dict_from_url(
                url,
                map_location="cpu",
                progress=True,
            )
        elif dino_depth is None:
            raise ValueError(
                "dino_depth is required when state_dict is provided "
                "(e.g. 12, 24, or 40 for ViT-g/14)."
            )

        start_idx = dino_depth - self.num_layers
        assert start_idx >= 0, (
            f"DINOv2 depth ({dino_depth}) too small for {self.num_layers} layer pairs "
            f"(need at least {self.num_layers} blocks)"
        )

        for i in range(self.num_layers):
            spatial_prefix = f"blocks.{start_idx + i}."
            spatial_sd = {
                k[len(spatial_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(spatial_prefix)
            }

            miss_s, unexp_s = self.spatial_blocks[i].load_state_dict(spatial_sd, strict=False)

            if miss_s:
                print(f"[load_dino_weights] spatial_blocks[{i}] missing keys: {miss_s}")

        norm_sd = {
            k[len("norm."):]: v
            for k, v in state_dict.items()
            if k.startswith("norm.") and not k.startswith("norm_")
        }
        if norm_sd:
            self.norm.load_state_dict(norm_sd)
            print("[load_dino_weights] Loaded DINOv2 final norm into norm")

    def forward(
        self,
        all_output: torch.Tensor,
        all_pos_embeds: torch.Tensor,
    ) -> tuple:
        """
        Args:
            static_output:          [N_s, C]
            dynamic_output:         [N_d, T, C]
            static_pos_embeds:      [N_s, C]
            dynamic_pos_embeds:     [N_d, T, C]
            temporal_cls_token_feat:[N_d, C]
            temporal_cls_token_pos: [N_d, C]

        Returns:
            static_output:          [N_s, C]
            dynamic_output:         [N_d, T, C]
            temporal_cls_token_feat:[N_d, C]
        """
        all_output = (
            all_output
            + all_pos_embeds
        )

        for i in range(self.num_layers):
            all_output = self.spatial_blocks[i](all_output[None]).squeeze(0)

        all_output = self.norm(all_output)
        return all_output
    
    

class TemporalPointAttentionsDino(nn.Module):
    """Spatio-temporal point attention built from DINOv2-compatible transformer blocks.

    Each of the ``num_layers`` iterations runs:
        1. A **spatial** DinoTransformerBlock over (static tokens + temporal cls tokens).
        2. A **temporal** DinoTransformerBlock over each dynamic point's time sequence
           (with the temporal cls token appended).

    Positional embeddings are added *once* before entering the layers (same
    pattern as SpatioTemporalPointAttentionsv2).

    Pretrained weights from the last ``num_layers * 2`` blocks of a DINOv2 ViT
    can be loaded via :meth:`load_dino_weights`.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        nheads: int = 6,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float = 1.0,
        dropout: float = 0.0,
        knn_spatial_attn: bool = False,
        k: int = 16,
        vit_name: str = "vits14",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.vit_name = vit_name
        
        if self.vit_name == "vits14":
            latent_dim = 384
            nheads = 6
        elif self.vit_name == "vitb14":
            latent_dim = 768
            nheads = 12
            num_layers = 3
            self.num_layers = 3
        elif self.vit_name == "vitl14":
            latent_dim = 1024
            nheads = 16
        elif self.vit_name == "vitg14":
            latent_dim = 1536
            nheads = 24
        
        block_kwargs = dict(
            dim=latent_dim,
            num_heads=nheads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
        )

        self.temporal_blocks = nn.ModuleList(
            [DinoTransformerBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self.spatial_cross_attn_blocks = nn.ModuleList(
            [DinoCrossTransformerBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(latent_dim, eps=1e-6)
        
        self.load_dino_weights()

    def load_dino_weights(
        self,
        state_dict: dict | None = None,
        dino_depth: int | None = None,
    ) -> None:
        """Load weights from a DINOv2 ViT state dict.

        The last ``num_layers * 2`` blocks are mapped to alternating
        spatial / temporal blocks::

            dino blocks[start + 2*i]     ->  temporal_blocks[i]
            dino blocks[start + 2*i + 1] ->  spatial_cross_attn_blocks[i]

        Args:
            state_dict: Full DINOv2 ViT state dict (keys like ``blocks.{i}.norm1.weight``),
                or ``None`` to download pretrained weights for ``self.vit_name`` (e.g.
                ``vits14`` or ``vitg14``) from the official DINOv2 URL.
            dino_depth: Total depth of the source ViT. Required when ``state_dict`` is
                provided (e.g. 12 for ViT-S/B, 24 for ViT-L, 40 for ViT-g/14). When
                ``state_dict`` is ``None``, depth follows the architecture (12 for
                ``vits14``, 40 for ``vitg14``).
        """
        if state_dict is None:
            try:
                url = _DINOV2_VIT_PRETRAIN_URL[self.vit_name]
                dino_depth = _DINOV2_VIT_DEPTH[self.vit_name]
            except KeyError as e:
                raise KeyError(
                    f"Unknown vit_name {self.vit_name!r}; expected one of "
                    f"{sorted(_DINOV2_VIT_PRETRAIN_URL.keys())}"
                ) from e
            state_dict = torch.hub.load_state_dict_from_url(
                url,
                map_location="cpu",
                progress=True,
            )
        elif dino_depth is None:
            raise ValueError(
                "dino_depth is required when state_dict is provided "
                "(e.g. 12, 24, or 40 for ViT-g/14)."
            )

        start_idx = dino_depth - self.num_layers * 2
        assert start_idx >= 0, (
            f"DINOv2 depth ({dino_depth}) too small for {self.num_layers} layer pairs "
            f"(need at least {self.num_layers * 2} blocks)"
        )

        for i in range(self.num_layers):
            temporal_prefix = f"blocks.{start_idx + 2 * i}."
            spatial_prefix = f"blocks.{start_idx + 2 * i + 1}."
            spatial_sd = {
                k[len(spatial_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(spatial_prefix)
            }
            temporal_sd = {
                k[len(temporal_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(temporal_prefix)
            }

            miss_t, unexp_t = self.temporal_blocks[i].load_state_dict(temporal_sd, strict=False)
            miss_s, unexp_s = self.spatial_cross_attn_blocks[i].load_dino_block_state_dict(spatial_sd, strict=False)

            if miss_s:
                print(f"[load_dino_weights] spatial_blocks[{i}] missing keys: {miss_s}")
            if miss_t:
                print(f"[load_dino_weights] temporal_blocks[{i}] missing keys: {miss_t}")

        norm_sd = {
            k[len("norm."):]: v
            for k, v in state_dict.items()
            if k.startswith("norm.") and not k.startswith("norm_")
        }
        if norm_sd:
            self.norm.load_state_dict(norm_sd)
            print("[load_dino_weights] Loaded DINOv2 final norm into norm")

    def forward(
        self,
        trajs: torch.Tensor,
        traj_pos_embeds: torch.Tensor,
        scene_context: torch.Tensor,
    ) -> tuple:
        """
        Args:
            trajs:                  [N_t, T, C]
            traj_pos_embeds:        [N_t, T, C]
            scene_context:          [N_s, C]

        Returns:
            trajs:                  [N_t, T, C]
        """
        if trajs.shape[0] == 0:
            return trajs
        
        trajs = trajs + traj_pos_embeds

        for i in range(self.num_layers):
            n_points = trajs.shape[0]
            max_chunk = 60000

            if n_points <= max_chunk:
                trajs = self.temporal_blocks[i](trajs)
            else:
                chunks = []
                for start in range(0, n_points, max_chunk):
                    end = min(start + max_chunk, n_points)
                    chunks.append(
                        self.temporal_blocks[i](trajs[start:end])
                    )
                trajs = torch.cat(chunks, dim=0)

            temporal_cls_token_feat = trajs[:, 0]

            # have attentions from cls tokens to scene context
            temporal_cls_token_feat = self.spatial_cross_attn_blocks[i](temporal_cls_token_feat[None], scene_context[None])
            trajs = torch.cat([temporal_cls_token_feat.permute(1, 0, 2), trajs[:, 1:]], dim=1)
            
  
        trajs = self.norm(trajs)

        return trajs

