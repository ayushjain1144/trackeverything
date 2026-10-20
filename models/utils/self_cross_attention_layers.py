# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from functools import partial
from typing import Optional

import ipdb

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

st = ipdb.set_trace


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        **kwargs,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            tgt2 = self.self_attn(
                q,
                k,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
            )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        **kwargs,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            tgt2 = self.self_attn(
                q,
                k,
                value=tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
            )[0]

        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        **kwargs,
    ):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            if self.normalize_before:
                return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
            return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        freeze_ffm_mlp=False,
        freeze_ffm_ln=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if freeze_ffm_mlp:
            for param in self.linear1.parameters():
                param.requires_grad = False
            for param in self.linear2.parameters():
                param.requires_grad = False

        self.norm = nn.LayerNorm(d_model)

        if freeze_ffm_ln:
            for param in self.norm.parameters():
                param.requires_grad = False

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class KNNSelfAttentionLayer(nn.Module):
    """Self-attention where each token attends only to its K nearest neighbors."""

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        k=16,
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.k = k
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _gather_knn(self, x, knn_indices):
        """
        x: (B, S, C)
        knn_indices: (B, S, K)
        returns: (B, S, K, C)
        """
        B, S, C = x.shape
        K = knn_indices.shape[-1]
        idx = knn_indices.reshape(B, S * K, 1).expand(-1, -1, C)
        return torch.gather(x, 1, idx).reshape(B, S, K, C)

    def _knn_self_attn(self, tgt, query_pos, knn_indices):
        """Each token attends to its K nearest neighbors via cross-attention.

        tgt: (S, B, C), query_pos: (S, B, C) or None, knn_indices: (B, S, K)
        """
        S, B, C = tgt.shape
        K = knn_indices.shape[-1]
        tgt_bsc = tgt.permute(1, 0, 2)  # (B, S, C)

        neighbors = self._gather_knn(tgt_bsc, knn_indices)  # (B, S, K, C)

        # Reshape: each token's neighborhood should become a sequence
        q = tgt_bsc.reshape(B * S, 1, C).permute(1, 0, 2)    # (1, B*S, C)
        kv = neighbors.reshape(B * S, K, C).permute(1, 0, 2)  # (K, B*S, C)

        q_pos, kv_pos = None, None
        if query_pos is not None:
            pos_bsc = query_pos.permute(1, 0, 2)
            nbr_pos = self._gather_knn(pos_bsc, knn_indices)
            q_pos = pos_bsc.reshape(B * S, 1, C).permute(1, 0, 2)
            kv_pos = nbr_pos.reshape(B * S, K, C).permute(1, 0, 2)

        new_bs = kv.shape[1]
        max_chunk = 60000
        # because flash attention only supports a maximum of 64K tokens
        if new_bs > max_chunk:
            chunks = []
            for start in range(0, new_bs, max_chunk):
                end = min(start + max_chunk, new_bs)
                chunks.append(self.self_attn(
                    self.with_pos_embed(q, q_pos)[:, start:end],
                    self.with_pos_embed(kv, kv_pos)[:, start:end],
                    value=kv[:, start:end],
                    need_weights=False,
                )[0])
            out = torch.cat(chunks, dim=1)
        else:
            out = self.self_attn(
                self.with_pos_embed(q, q_pos),
                self.with_pos_embed(kv, kv_pos),
                value=kv,
                need_weights=False,
            )[0]

        return out.squeeze(0).reshape(B, S, C).permute(1, 0, 2)  # (S, B, C)

    def forward_post(self, tgt, query_pos, knn_indices):
        tgt2 = self._knn_self_attn(tgt, query_pos, knn_indices)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, query_pos, knn_indices):
        tgt2 = self.norm(tgt)
        tgt2 = self._knn_self_attn(tgt2, query_pos, knn_indices)
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        knn_indices: Optional[Tensor] = None,
        **kwargs,
    ):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            if self.normalize_before:
                return self.forward_pre(tgt, query_pos, knn_indices)
            return self.forward_post(tgt, query_pos, knn_indices)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    Used for nn.Transformer that uses a HW x N x C rep
    """

    def forward(self, x):
        """
        x: HW x N x C
        permute to N x C x HW
        Apply BN on C
        permute back
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        # x: n x c x hw -> hw x n x c
        x = x.permute(2, 0, 1)
        return x


NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


